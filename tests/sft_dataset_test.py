#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic CPU-only unit tests for Stage C SFT & Preference dataset pack.

Usage: python3 -I tests/sft_dataset_test.py
No GPU, real model, optimizer, or resident serving process is used.
"""

import base64
from contextlib import contextmanager
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import narrative_rl_fixtures as fixture  # noqa: E402
import umminfer_rl_reward as reward_mod  # noqa: E402
import umminfer_rl_trajectory as traj  # noqa: E402
import umminfer_sft_dataset as ds  # noqa: E402


class _CannedHandler(BaseHTTPRequestHandler):
    response_bytes = b""
    content_type = "text/event-stream"

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", self.content_type)
        self.send_header("Content-Length", str(len(self.response_bytes)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(self.response_bytes)
        self.close_connection = True


@contextmanager
def _canned_server(response_bytes, content_type="text/event-stream"):
    handler = type("Handler", (_CannedHandler,), {
        "response_bytes": response_bytes, "content_type": content_type,
    })
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _data_url(mime, raw):
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def _tree_bytes(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(Path(root).rglob("*"))
        if path.is_file()
    }


def _make_v2_record(recorder, *, prompt_text, clean_parts, slots=(),
                    session_id="sess-0", request_id="req-0",
                    expected_modalities=None, include_media_inputs=False,
                    terminal_state="completed", error=None):
    """Record a schema-v2 trajectory directly via TrajectoryRecorder."""
    inputs = [{
        "kind": "text",
        "ref": None,
        "media_type": None,
        "sha256": traj.sha256_text(prompt_text),
        "text": prompt_text,
    }]
    if include_media_inputs:
        img_sha = traj.sha256_bytes(fixture.PNG)
        recorder.artifact_dir.mkdir(parents=True, exist_ok=True)
        img_name = img_sha + ".png"
        (recorder.artifact_dir / img_name).write_bytes(fixture.PNG)
        inputs.append({
            "kind": "image",
            "ref": img_name,
            "media_type": "image/png",
            "sha256": img_sha,
            "bytes": len(fixture.PNG),
        })

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
            "state": "completed" if terminal_state == "completed" else "aborted",
        })

    output_text = "".join(clean_parts)
    term_kind = traj.TERMINAL_EVENT_FOR_STATE[terminal_state]
    term_event = {
        "seq": seq,
        "server_seq": seq,
        "type": term_kind,
        "stage": None,
        "is_final": True,
        "text": output_text if term_kind == "complete" else (error or ""),
        "legs": legs,
    }
    if term_kind == "error":
        term_event["error"] = error or "synthetic error"
    events.append(term_event)

    record = recorder.record(
        session_id=session_id,
        request_id=request_id,
        endpoint="/v1/chat/completions",
        inputs=inputs,
        params={"model": "fixture-model", "stream": True},
        events=events,
        terminal_state=terminal_state,
        error=error if terminal_state == "failed" else None,
        output_text=output_text,
        artifacts=artifacts,
    )
    if expected_modalities is not None:
        # Rewrite the last JSONL line with out-of-band ablation label metadata.
        lines = recorder.jsonl_path.read_text(encoding="utf-8").splitlines()
        enriched = json.loads(lines[-1])
        enriched["ablation"] = {
            "expected_visual_modalities": list(expected_modalities),
        }
        lines[-1] = ds.canonical_json(enriched).decode("utf-8")
        recorder.jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return enriched
    return record


class _LengthSlotReward(reward_mod.Reward):
    """Export-only test reward for pluggable module:attr verification."""

    name = "test-length-plus-slots-v1"

    def score(self, record):
        text = record.get("clean_text", record.get("output_text", ""))
        slots = record.get("visual_slots", [])
        return float(len(text) / 100.0 + 0.5 * len(slots))


class _NonFiniteReward(reward_mod.Reward):
    """Export-only test reward returning invalid/non-finite values."""

    name = "test-non-finite-v1"

    def score(self, record):
        text = record.get("clean_text", "")
        if "nan" in text:
            return float("nan")
        if "inf" in text:
            return float("inf")
        if "bool" in text:
            return True
        return 1.0


class SftDatasetBuilderTests(unittest.TestCase):
    def test_sft_interleaved_reconstruction_dedup_and_byte_determinism(self):
        with tempfile.TemporaryDirectory(prefix="umm-sft-det-") as tmp:
            root = Path(tmp)
            rec_path = root / "rollouts.jsonl"
            recorder = traj.TrajectoryRecorder(rec_path)

            body = {
                "model": "narrative-fixture",
                "stream": True,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "tell the lantern story"},
                    {"type": "input_audio", "input_audio": {
                        "format": "wav",
                        "data": base64.b64encode(fixture.WAV).decode(),
                    }},
                    {"type": "image_url", "image_url": {
                        "url": _data_url("image/png", fixture.PNG),
                    }},
                ]}],
                "narrative": {"enabled": True, "visual_mode": "auto", "max_slots": 3},
                "modalities": ["text", "image", "audio", "video"],
                "image": {"steps": 4, "image": _data_url("image/png", fixture.PNG)},
                "audio": {"voice": "vivian", "stream": True,
                          "ref_audio": base64.b64encode(fixture.WAV).decode()},
                "video": {"num_frames": 13, "steps": 4},
                "seed": 42,
            }
            sse = fixture.sse_bytes(fixture.narrative_events())
            with _canned_server(sse) as base_url:
                # Record the same turn twice to exercise (prompt, target) dedup.
                traj.record_chat_completion(
                    base_url, copy.deepcopy(body), recorder, "sess-a")
                traj.record_chat_completion(
                    base_url, copy.deepcopy(body), recorder, "sess-b")

            # Add a text-only completed v2 turn.
            _make_v2_record(
                recorder,
                prompt_text="Explain conservation of momentum concisely.",
                clean_parts=["Total momentum in an isolated system is constant."],
                slots=(),
                session_id="sess-text",
                request_id="req-text",
            )
            # Add a UTF-8 multi-byte narrative turn to verify byte-offset slicing.
            utf8_first = "黎明时分，"
            utf8_second = "一盏红灯笼缓缓升起。"
            _make_v2_record(
                recorder,
                prompt_text="用中文讲述红灯笼的故事",
                clean_parts=[utf8_first, utf8_second],
                slots=[("image", "glowing red lantern at dawn",
                        len(utf8_first.encode("utf-8")))],
                session_id="sess-utf8",
                request_id="req-utf8",
            )

            input_bytes_before = rec_path.read_bytes()
            out_a = root / "pack_a"
            out_b = root / "pack_b"
            manifest_a = ds.build_dataset(
                [rec_path], out_a, mode="sft", seed="det-seed", eval_ratio=0.5)
            manifest_b = ds.build_dataset(
                [rec_path], out_b, mode="sft", seed="det-seed", eval_ratio=0.5)

            # Input file remained strictly read-only.
            self.assertEqual(rec_path.read_bytes(), input_bytes_before)
            # Output trees are byte-for-byte identical across independent runs.
            self.assertEqual(_tree_bytes(out_a), _tree_bytes(out_b))
            self.assertEqual(manifest_a, manifest_b)
            self.assertEqual(manifest_a["REAL_TRAINING"], "NOT_RUN")
            self.assertEqual(manifest_a["counts"]["total_records_read"], 4)
            self.assertEqual(manifest_a["counts"]["accepted_verified_completed"], 4)
            self.assertEqual(manifest_a["counts"]["rejected"]["duplicate"], 1)
            self.assertEqual(manifest_a["split_audit"]["leakage_count"], 0)
            self.assertEqual(
                manifest_a["artifact_verification"]["mode"], "verified_on_disk")

            # Also verify permutation invariance: reverse input lines -> identical pack.
            rev_path = root / "rollouts_rev.jsonl"
            rev_lines = list(reversed(rec_path.read_text(encoding="utf-8").splitlines()))
            rev_path.write_text("\n".join(rev_lines) + "\n", encoding="utf-8")
            out_rev = root / "pack_rev"
            ds.build_dataset(
                [rev_path], out_rev, mode="sft",
                artifact_dir=recorder.artifact_dir,
                seed="det-seed", eval_ratio=0.5)
            self.assertEqual(
                (out_a / "sft_train.jsonl").read_bytes(),
                (out_rev / "sft_train.jsonl").read_bytes())
            self.assertEqual(
                (out_a / "sft_eval.jsonl").read_bytes(),
                (out_rev / "sft_eval.jsonl").read_bytes())

            # Inspect reconstructed targets across train+eval.
            rows = []
            for fname in ("sft_train.jsonl", "sft_eval.jsonl"):
                for line in (out_a / fname).read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rows.append(json.loads(line))
            self.assertEqual(len(rows), 3)

            by_prompt = {row["prompt_text"]: row for row in rows}
            lantern = by_prompt["tell the lantern story"]
            expected_target = (
                "At dawn, "
                "[[image: a glowing red lantern]]"
                "[[video: the lantern drifts downstream]]"
                "a red lantern rises."
                "[[image: an optional alternate view]]"
            )
            self.assertEqual(lantern["assistant_target"], expected_target)
            self.assertEqual(lantern["clean_text"], fixture.CLEAN_TEXT)
            self.assertEqual(lantern["modality_route"], ["image", "video", "image"])
            self.assertFalse(lantern["is_text_only"])
            self.assertEqual(lantern["visual_slots"][0]["leg_state"], "completed")
            self.assertEqual(lantern["visual_slots"][2]["leg_state"], "failed")
            self.assertEqual(
                lantern["visual_slots"][0]["artifact_sha256"],
                traj.sha256_bytes(fixture.PNG))
            self.assertTrue(
                all(ref["verified_on_disk"] for ref in lantern["input_media_refs"]))
            self.assertTrue(
                all(ref["verified_on_disk"] for ref in lantern["output_media_refs"]))
            # Zero embedded base64 media payloads in dataset JSONL.
            serialized = json.dumps(lantern)
            self.assertNotIn("data:image/", serialized)
            self.assertNotIn("data:audio/", serialized)

            text_only = by_prompt["Explain conservation of momentum concisely."]
            self.assertTrue(text_only["is_text_only"])
            self.assertEqual(
                text_only["assistant_target"],
                "Total momentum in an isolated system is constant.")
            self.assertNotIn("[[", text_only["assistant_target"])

            utf8_row = by_prompt["用中文讲述红灯笼的故事"]
            self.assertEqual(
                utf8_row["assistant_target"],
                "黎明时分，[[image: glowing red lantern at dawn]]一盏红灯笼缓缓升起。")

            audit = ds.audit_dataset_pack(out_a)
            self.assertEqual(audit["status"], "PASSED")
            self.assertEqual(audit["split_audit"]["leakage_count"], 0)

    def test_filters_cancelled_failed_tampered_hash_and_tampered_artifact(self):
        with tempfile.TemporaryDirectory(prefix="umm-sft-filter-") as tmp:
            root = Path(tmp)
            rec_path = root / "mixed.jsonl"
            recorder = traj.TrajectoryRecorder(rec_path)

            # 1) Valid completed turn (kept)
            _make_v2_record(
                recorder,
                prompt_text="valid prompt",
                clean_parts=["clean answer"],
                slots=(),
                session_id="s-ok",
                request_id="r-ok",
            )
            # 2) Valid cancelled turn (filtered as cancelled)
            with _canned_server(fixture.sse_bytes(fixture.cancelled_events())) as url:
                traj.record_chat_completion(
                    url,
                    {"model": "m", "stream": True,
                     "messages": [{"role": "user", "content": "cancelled prompt"}]},
                    recorder, "s-cancel")
            # 3) Valid failed turn (filtered as failed)
            _make_v2_record(
                recorder,
                prompt_text="failed prompt",
                clean_parts=[],
                slots=(),
                session_id="s-fail",
                request_id="r-fail",
                terminal_state="failed",
                error="upstream timeout",
            )
            # 4) Tampered output_sha256 record (filtered as unverified)
            valid_for_tamper = _make_v2_record(
                recorder,
                prompt_text="tampered hash prompt",
                clean_parts=["original"],
                slots=(),
                session_id="s-tamp",
                request_id="r-tamp",
            )
            # 5) Turn with a CAS image whose bytes we will tamper on disk
            tampered_art_rec = _make_v2_record(
                recorder,
                prompt_text="tampered artifact prompt",
                clean_parts=["with image"],
                slots=[("image", "sunset over ocean", 10)],
                session_id="s-art",
                request_id="r-art",
            )

            # Tamper record #4 hash in JSONL, plus append an invalid JSON line.
            lines = rec_path.read_text(encoding="utf-8").splitlines()
            bad_hash = json.loads(lines[3])
            self.assertEqual(bad_hash["request_id"], valid_for_tamper["request_id"])
            bad_hash["output_sha256"] = "0" * 64
            lines[3] = ds.canonical_json(bad_hash).decode("utf-8")
            lines.append("{not valid json")
            rec_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            # Tamper CAS artifact bytes on disk for record #5.
            art_ref = tampered_art_rec["artifacts"][0]["ref"]
            art_path = recorder.artifact_dir / art_ref
            art_path.write_bytes(b"CORRUPTED_BYTES_" + b"x" * 16)

            out_dir = root / "out"
            manifest = ds.build_dataset([rec_path], out_dir, mode="sft", eval_ratio=0.0)
            rejected = manifest["counts"]["rejected"]
            self.assertEqual(manifest["counts"]["total_records_read"], 6)
            self.assertEqual(manifest["counts"]["accepted_verified_completed"], 1)
            self.assertEqual(rejected["cancelled"], 1)
            self.assertEqual(rejected["failed"], 1)
            self.assertEqual(rejected["unverified"], 3)  # bad hash + tampered CAS + bad JSON
            self.assertEqual(manifest["counts"]["rejection_details"]["artifact_tampered"], 1)
            self.assertEqual(manifest["counts"]["rejection_details"]["integrity_error"], 1)
            self.assertEqual(manifest["counts"]["rejection_details"]["parse_error"], 1)
            self.assertEqual(manifest["outputs"]["sft_train"]["record_count"], 1)

            only_row = json.loads((out_dir / "sft_train.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(only_row["prompt_text"], "valid prompt")

    def test_preference_dpo_and_grpo_with_default_and_custom_rewards(self):
        with tempfile.TemporaryDirectory(prefix="umm-pref-") as tmp:
            root = Path(tmp)
            rec_path = root / "ablation_rollouts.jsonl"
            recorder = traj.TrajectoryRecorder(rec_path)

            # Prompt 1 expects ["image", "video"]:
            # - rollout A: predicts ["image", "video"] -> F1=1.0, red=0.0 -> score=1.0
            # - rollout B: predicts ["image"]          -> F1=2/3, red=0.0 -> score=2/3
            # - rollout C: predicts []                 -> F1=0.0, red=0.0 -> score=0.0
            # - rollout D: predicts ["image", "image"] -> F1=2/3, red=0.5 -> score=1/6
            p1 = "Explain black hole accretion disks with visual beats."
            _make_v2_record(
                recorder, prompt_text=p1,
                clean_parts=["Accretion disks glow intensely."],
                slots=[("image", "accretion disk diagram", 15),
                       ("video", "matter spiraling into event horizon", 31)],
                session_id="p1-a", request_id="p1-a",
                expected_modalities=["image", "video"],
            )
            _make_v2_record(
                recorder, prompt_text=p1,
                clean_parts=["Accretion disks glow intensely."],
                slots=[("image", "accretion disk diagram", 15)],
                session_id="p1-b", request_id="p1-b",
                expected_modalities=["image", "video"],
            )
            _make_v2_record(
                recorder, prompt_text=p1,
                clean_parts=["Accretion disks glow intensely."],
                slots=(),
                session_id="p1-c", request_id="p1-c",
                expected_modalities=["image", "video"],
            )
            _make_v2_record(
                recorder, prompt_text=p1,
                clean_parts=["Accretion disks glow intensely."],
                slots=[("image", "disk view one", 10),
                       ("image", "disk view two", 31)],
                session_id="p1-d", request_id="p1-d",
                expected_modalities=["image", "video"],
            )

            # Prompt 2: zero-variance group (both expect [] and predict [], score=1.0)
            p2 = "What is 17 * 23?"
            _make_v2_record(
                recorder, prompt_text=p2, clean_parts=["391."],
                slots=(), session_id="p2-a", request_id="p2-a",
                expected_modalities=[],
            )
            _make_v2_record(
                recorder, prompt_text=p2, clean_parts=["The product is 391."],
                slots=(), session_id="p2-b", request_id="p2-b",
                expected_modalities=[],
            )

            # Prompt 3: unlabelled record -> NarrativeRoutingReward returns None (unscored)
            _make_v2_record(
                recorder, prompt_text="Unlabelled prompt",
                clean_parts=["Some text."], slots=(),
                session_id="p3-a", request_id="p3-a",
                expected_modalities=None,
            )

            # Prompt 4: singleton labelled rollout -> skipped_small_groups=1
            _make_v2_record(
                recorder, prompt_text="Singleton prompt",
                clean_parts=["Only one rollout."], slots=(),
                session_id="p4-a", request_id="p4-a",
                expected_modalities=[],
            )

            out_dir = root / "pref_pack"
            manifest = ds.build_dataset(
                [rec_path], out_dir,
                mode="preference",
                seed="pref-seed",
                eval_ratio=0.0,
                min_margin=0.4,
            )
            self.assertEqual(manifest["REAL_TRAINING"], "NOT_RUN")
            self.assertEqual(
                manifest["reward"]["name"],
                "narrative-routing-f1-minus-redundancy-v1")
            self.assertEqual(manifest["counts"]["rejected"]["unscored"], 1)
            self.assertEqual(manifest["counts"]["skipped_small_groups"], 1)

            grpo_rows = [
                json.loads(line)
                for line in (out_dir / "grpo_train.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(grpo_rows), 2)
            grpo_by_prompt = {row["prompt_text"]: row for row in grpo_rows}

            g1 = grpo_by_prompt[p1]
            self.assertEqual(g1["group_size"], 4)
            self.assertTrue(g1["has_reward_variance"])
            scores = [r["score"] for r in g1["rollouts"]]
            self.assertEqual(scores, sorted(scores, reverse=True))
            advs = [r["advantage"] for r in g1["rollouts"]]
            self.assertAlmostEqual(sum(advs) / len(advs), 0.0, places=9)
            std_adv = math.sqrt(sum(a * a for a in advs) / len(advs))
            self.assertAlmostEqual(std_adv, 1.0, places=9)
            self.assertGreater(advs[0], 0.0)
            self.assertLess(advs[-1], 0.0)

            g2 = grpo_by_prompt[p2]
            self.assertFalse(g2["has_reward_variance"])
            self.assertEqual([r["advantage"] for r in g2["rollouts"]], [0.0, 0.0])

            # With scores [1.0, 2/3, 1/6, 0.0] and min_margin=0.4:
            # Valid pairs with margin >= 0.4:
            # (1.0, 1/6)=0.833, (1.0, 0.0)=1.0, (2/3, 1/6)=0.5, (2/3, 0.0)=0.667 -> 4 pairs.
            # Pairs < 0.4: (1.0, 2/3)=0.333, (1/6, 0.0)=0.167, plus tied p2 (1.0, 1.0)=0.0 -> 3 skipped.
            dpo_rows = [
                json.loads(line)
                for line in (out_dir / "dpo_train.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(dpo_rows), 4)
            self.assertEqual(manifest["counts"]["skipped_low_margin_pairs"], 3)
            for pair in dpo_rows:
                self.assertGreaterEqual(pair["margin"] + 1e-12, 0.4)
                self.assertGreater(pair["chosen"]["score"], pair["rejected"]["score"])
                self.assertNotEqual(
                    pair["chosen"]["target_sha256"],
                    pair["rejected"]["target_sha256"])

            # Verify pluggable custom reward via load_reward("module:attr").
            out_custom = root / "pref_custom"
            custom_manifest = ds.build_dataset(
                [rec_path], out_custom,
                mode="preference",
                reward_spec="sft_dataset_test:_LengthSlotReward",
                eval_ratio=0.0,
                min_margin=0.01,
            )
            self.assertEqual(
                custom_manifest["reward"]["name"], "test-length-plus-slots-v1")
            self.assertEqual(custom_manifest["counts"]["rejected"]["unscored"], 0)
            self.assertEqual(ds.audit_dataset_pack(out_custom)["status"], "PASSED")

            # Verify preference pack is byte-identical across two independent runs.
            out_pref_b = root / "pref_pack_b"
            ds.build_dataset(
                [rec_path], out_pref_b,
                mode="preference",
                seed="pref-seed",
                eval_ratio=0.0,
                min_margin=0.4,
            )
            self.assertEqual(_tree_bytes(out_dir), _tree_bytes(out_pref_b))

    def test_missing_cas_artifact_slot_tampering_and_non_finite_rewards(self):
        with tempfile.TemporaryDirectory(prefix="umm-sft-edge-") as tmp:
            root = Path(tmp)
            rec_path = root / "edge.jsonl"
            recorder = traj.TrajectoryRecorder(rec_path)

            base_ok = _make_v2_record(
                recorder, prompt_text="good image turn",
                clean_parts=["hello world"],
                slots=[("image", "mountain peak", 5)],
                session_id="e-ok", request_id="e-ok",
            )
            missing_art = _make_v2_record(
                recorder, prompt_text="missing cas turn",
                clean_parts=["missing asset"],
                slots=[("video", "ocean waves", 7)],
                session_id="e-miss", request_id="e-miss",
            )
            # Remove the video artifact file so CAS check catches missing file.
            (recorder.artifact_dir / missing_art["artifacts"][0]["ref"]).unlink()

            # Append structural tamper variants derived from base_ok.
            lines = rec_path.read_text(encoding="utf-8").splitlines()
            mutations = [
                lambda r: r["visual_slots"][0].__setitem__("slot_index", 5),
                lambda r: r.__setitem__("clean_text", r["clean_text"] + "!"),
                lambda r: r["events"][2]["artifact"].__setitem__("data", "AA=="),
                lambda r: r["events"][2]["artifact"].__setitem__("chunk_sha256", "0" * 64),
                lambda r: r.__setitem__("schema_version", True),
            ]
            for idx, mutate in enumerate(mutations):
                bad = copy.deepcopy(base_ok)
                bad["request_id"] = f"tampered-{idx}"
                mutate(bad)
                lines.append(ds.canonical_json(bad).decode("utf-8"))
            rec_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            out_sft = root / "edge_sft"
            manifest = ds.build_dataset([rec_path], out_sft, mode="sft", eval_ratio=0.0)
            self.assertEqual(manifest["counts"]["accepted_verified_completed"], 1)
            self.assertEqual(manifest["counts"]["rejected"]["unverified"], 6)
            self.assertEqual(
                manifest["counts"]["rejection_details"]["artifact_missing"], 1)
            self.assertEqual(
                manifest["counts"]["rejection_details"]["integrity_error"], 5)

            # Verify non-finite/bool rewards are rejected cleanly without writing NaN/Inf.
            nf_path = root / "nonfinite.jsonl"
            nf_rec = traj.TrajectoryRecorder(nf_path)
            for idx, text in enumerate(("nan case", "inf case", "bool case", "ok a", "ok b")):
                _make_v2_record(
                    nf_rec, prompt_text="shared prompt",
                    clean_parts=[text], slots=(),
                    session_id=f"nf-{idx}", request_id=f"nf-{idx}",
                )
            out_nf = root / "edge_nf"
            nf_manifest = ds.build_dataset(
                [nf_path], out_nf,
                mode="preference",
                reward_spec="sft_dataset_test:_NonFiniteReward",
                eval_ratio=0.0,
            )
            self.assertEqual(
                nf_manifest["counts"]["rejection_details"]["invalid_reward"], 3)
            self.assertEqual(nf_manifest["outputs"]["grpo_train"]["record_count"], 1)
            self.assertEqual(ds.audit_dataset_pack(out_nf)["status"], "PASSED")

    def test_zero_prompt_leakage_across_splits_and_tamper_detection_in_auditor(self):
        with tempfile.TemporaryDirectory(prefix="umm-split-audit-") as tmp:
            root = Path(tmp)
            rec_path = root / "many_prompts.jsonl"
            recorder = traj.TrajectoryRecorder(rec_path)

            for idx in range(25):
                prompt = f"Prompt topic #{idx}: illustrate key dynamics."
                # Create 2 distinct rollouts per prompt so every prompt has
                # multiple examples across SFT, DPO, and GRPO.
                _make_v2_record(
                    recorder, prompt_text=prompt,
                    clean_parts=[f"Detailed explanation {idx}."],
                    slots=[("image", f"diagram for topic {idx}", 10)],
                    session_id=f"s-{idx}-a", request_id=f"r-{idx}-a",
                    expected_modalities=["image"],
                )
                _make_v2_record(
                    recorder, prompt_text=prompt,
                    clean_parts=[f"Brief text-only explanation {idx}."],
                    slots=(),
                    session_id=f"s-{idx}-b", request_id=f"r-{idx}-b",
                    expected_modalities=["image"],
                )

            out_both = root / "both_pack"
            manifest = ds.build_dataset(
                [rec_path], out_both,
                mode="both",
                seed="split-audit-seed-42",
                eval_ratio=0.3,
                min_margin=0.1,
            )
            split_audit = manifest["split_audit"]
            self.assertEqual(split_audit["leakage_count"], 0)
            self.assertTrue(split_audit["passed"])
            self.assertGreater(split_audit["train_prompt_count"], 0)
            self.assertGreater(split_audit["eval_prompt_count"], 0)
            self.assertEqual(
                split_audit["train_prompt_count"] + split_audit["eval_prompt_count"],
                25)

            # Verify identical prompt split assignment across sft, dpo, and grpo.
            for pack_name in ("sft", "dpo", "grpo"):
                self.assertEqual(split_audit["packs"][pack_name]["leakage_count"], 0)
                self.assertEqual(
                    split_audit["packs"][pack_name]["train_prompt_count"],
                    split_audit["train_prompt_count"])
                self.assertEqual(
                    split_audit["packs"][pack_name]["eval_prompt_count"],
                    split_audit["eval_prompt_count"])

            self.assertEqual(ds.audit_dataset_pack(out_both)["status"], "PASSED")

            # Tampering an output JSONL file is caught by audit_dataset_pack.
            tampered_pack = root / "tampered_jsonl"
            ds.build_dataset([rec_path], tampered_pack, mode="sft",
                             seed="split-audit-seed-42", eval_ratio=0.3)
            train_file = tampered_pack / "sft_train.jsonl"
            train_file.write_bytes(train_file.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                ds.audit_dataset_pack(tampered_pack)

            # Tampering manifest.json without sidecar update is caught.
            tampered_manifest = root / "tampered_manifest"
            ds.build_dataset([rec_path], tampered_manifest, mode="sft",
                             seed="split-audit-seed-42", eval_ratio=0.3)
            mpath = tampered_manifest / "manifest.json"
            mobj = json.loads(mpath.read_text(encoding="utf-8"))
            mobj["seed"] = "other"
            mpath.write_bytes(ds.canonical_json(mobj) + b"\n")
            with self.assertRaisesRegex(ValueError, "manifest.json.sha256 mismatch"):
                ds.audit_dataset_pack(tampered_manifest)

            # Direct leakage audit catches cross-split prompt overlap.
            leak = ds.audit_split_leakage(
                [{"prompt_sha256": "a" * 64}],
                [{"prompt_sha256": "a" * 64}, {"prompt_sha256": "b" * 64}],
            )
            self.assertEqual(leak["leakage_count"], 1)
            self.assertFalse(leak["passed"])
            self.assertEqual(leak["overlap_prompt_sha256s"], ["a" * 64])

    def test_mixed_schema_v1_and_v2_support_and_cli_execution(self):
        with tempfile.TemporaryDirectory(prefix="umm-v1v2-cli-") as tmp:
            root = Path(tmp)
            v1_path = root / "v1.jsonl"
            v1_record = {
                "schema_version": 1,
                "session_id": "legacy-sess",
                "request_id": "legacy-req",
                "turn_index": 0,
                "endpoint": "/v1/chat/completions",
                "inputs": [{
                    "kind": "text", "ref": None, "media_type": None,
                    "sha256": traj.sha256_text("legacy prompt"),
                    "text": "legacy prompt",
                }],
                "request_params": {"stream": True},
                "events": [{"seq": 0, "type": "output_delta", "text": "legacy answer"}],
                "terminal_state": "completed",
                "error": None,
                "output_text": "legacy answer",
                "output_sha256": traj.sha256_text("legacy answer"),
            }
            v1_path.write_text(
                ds.canonical_json(v1_record).decode("utf-8") + "\n",
                encoding="utf-8")

            v2_path = root / "v2.jsonl"
            recorder = traj.TrajectoryRecorder(v2_path)
            _make_v2_record(
                recorder, prompt_text="v2 prompt",
                clean_parts=["v2 answer"],
                slots=[("video", "orbiting planets", 9)],
                session_id="v2-sess", request_id="v2-req",
                include_media_inputs=True,
            )

            out_cli = root / "cli_out"
            cmd = [
                sys.executable, "-I",
                str(ROOT / "python" / "umminfer_sft_dataset.py"),
                "--input", str(v1_path), str(v2_path),
                "--out", str(out_cli),
                "--mode", "sft",
                "--seed", "cli-seed",
                "--eval-ratio", "0.0",
            ]
            completed = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            manifest = json.loads(completed.stdout)
            self.assertEqual(manifest["REAL_TRAINING"], "NOT_RUN")
            self.assertEqual(manifest["outputs"]["sft_train"]["record_count"], 2)

            audit_cmd = [
                sys.executable, "-I",
                str(ROOT / "python" / "umminfer_sft_dataset.py"),
                "--audit-only", "--out", str(out_cli),
            ]
            audited = subprocess.run(audit_cmd, capture_output=True, text=True, timeout=60)
            self.assertEqual(audited.returncode, 0, audited.stdout + audited.stderr)
            audit_payload = json.loads(audited.stdout)
            self.assertEqual(audit_payload["status"], "PASSED")
            self.assertEqual(audit_payload["REAL_TRAINING"], "NOT_RUN")

    def test_explicit_anchor_reconstruction_and_integrity(self):
        with tempfile.TemporaryDirectory(prefix="umm-sft-linked-") as tmp:
            recorder = traj.TrajectoryRecorder(Path(tmp) / "linked.jsonl")
            body = {"model": "m", "stream": True,
                    "messages": [{"role": "user", "content": "animate the lantern"}]}
            with _canned_server(fixture.sse_bytes(fixture.linked_events())) as url:
                record = traj.record_chat_completion(url, body, recorder, "linked")
            ok, category, message, _ = ds.verify_record_integrity(
                record, artifact_dir=recorder.artifact_dir,
                require_artifact_dir=True)
            self.assertTrue(ok, (category, message))
            target, clean, slots, media_refs = ds.reconstruct_director_target(
                record, verified_on_disk=True)
            self.assertEqual(clean, fixture.CLEAN_TEXT)
            self.assertEqual(
                target,
                fixture.FIRST_TEXT
                + "[[image: a glowing red lantern]]"
                + "[[video: the lantern sways in the wind | source=slot-0]]"
                + fixture.SECOND_TEXT)
            self.assertEqual(slots[1]["anchor_slot_index"], 0)
            self.assertTrue(slots[1]["anchor_explicit"])
            self.assertEqual(slots[1]["anchor_sha256"],
                             hashlib.sha256(fixture.PNG).hexdigest())
            self.assertTrue(all(ref["verified_on_disk"] for ref in media_refs))
            self.assertNotIn("data:image/", json.dumps(media_refs))

            bad = copy.deepcopy(record)
            for leg in bad["terminal_legs"]:
                if leg.get("slot_id") == "slot-1":
                    leg["anchor_sha256"] = "0" * 64
            bad["events"][-1]["legs"] = copy.deepcopy(bad["terminal_legs"])
            ok, _, message, _ = ds.verify_record_integrity(
                bad, artifact_dir=recorder.artifact_dir,
                require_artifact_dir=True)
            self.assertFalse(ok)
            self.assertIn("anchor", message)

            def remove_video(candidate):
                candidate["events"] = [event for event in candidate["events"]
                                       if not (event.get("type") == "artifact"
                                               and event.get("stage") == "video")]
                for seq, event in enumerate(candidate["events"]):
                    event["seq"] = seq
                    if event.get("server_seq") is not None:
                        event["server_seq"] = seq
                candidate["terminal_event_seq"] = len(candidate["events"]) - 1
                candidate["artifacts"] = [item for item in candidate["artifacts"]
                                          if item["media_type"] != "video"]

            missing = copy.deepcopy(record)
            remove_video(missing)
            ok, _, message, _ = ds.verify_record_integrity(
                missing, artifact_dir=recorder.artifact_dir,
                require_artifact_dir=True)
            self.assertFalse(ok)
            self.assertIn("no artifact", message)

            unresolved = copy.deepcopy(record)
            remove_video(unresolved)
            unresolved["visual_slots"][1].pop("anchor_slot_index")
            unresolved["visual_slots"][1]["prompt"] = ""
            for event in unresolved["events"]:
                if event.get("type") == "visual_slot" \
                        and event.get("slot_id") == "slot-1":
                    event.pop("anchor_slot_index")
                    event["prompt"] = ""
            for leg in unresolved["terminal_legs"]:
                if leg.get("slot_id") == "slot-1":
                    leg["state"] = "failed"
                    leg.pop("anchor_slot_index")
                    leg.pop("anchor_sha256")
            unresolved["events"][-1]["legs"] = copy.deepcopy(
                unresolved["terminal_legs"])
            ok, category, message, _ = ds.verify_record_integrity(
                unresolved, artifact_dir=recorder.artifact_dir,
                require_artifact_dir=True)
            self.assertTrue(ok, (category, message))
            target, _, _, _ = ds.reconstruct_director_target(
                unresolved, verified_on_disk=True)
            self.assertEqual(
                target,
                fixture.FIRST_TEXT
                + "[[image: a glowing red lantern]]"
                + fixture.SECOND_TEXT)

            legacy = copy.deepcopy(record)
            for slot in legacy["visual_slots"]:
                for key in ("continuity_mode", "anchor_slot_index", "anchor_explicit"):
                    slot.pop(key, None)
            for event in legacy["events"]:
                if event.get("type") == "visual_slot" \
                        and event.get("slot_id") == "slot-1":
                    event.pop("anchor_explicit")
                if event.get("type") == "artifact" \
                        and event.get("stage") == "video":
                    for key in ("anchor_slot_index", "anchor_explicit", "anchor_sha256"):
                        event["artifact"].pop(key, None)
            for leg in legacy["terminal_legs"]:
                if leg.get("slot_id") == "slot-1":
                    for key in ("anchor_slot_index", "anchor_explicit", "anchor_sha256"):
                        leg.pop(key, None)
            legacy["events"][-1]["legs"] = copy.deepcopy(legacy["terminal_legs"])
            for output in legacy["artifacts"]:
                if output["media_type"] == "video":
                    for key in ("anchor_slot_index", "anchor_explicit", "anchor_sha256"):
                        output.pop(key, None)
            ok, category, message, _ = ds.verify_record_integrity(
                legacy, artifact_dir=recorder.artifact_dir,
                require_artifact_dir=True)
            self.assertTrue(ok, (category, message))

    def test_serving_path_isolation_and_zero_ml_framework_dependencies(self):
        for directory in (ROOT / "src", ROOT / "include"):
            if not directory.is_dir():
                continue
            for path in directory.rglob("*"):
                if path.is_file():
                    self.assertNotIn(
                        "umminfer_sft_dataset",
                        path.read_text(encoding="utf-8", errors="ignore"),
                        str(path),
                    )
        # Also verify no serving Python runtime imports umminfer_sft_dataset.
        # The bounded optimizer adapters and online orchestrator are explicit
        # offline consumers; their preflights forbid ML imports until validation.
        offline_dataset_consumers = {
            "umminfer_sft_dataset.py",
            "umminfer_joint_sft_live.py",
            "umminfer_joint_dpo_live.py",
            "umminfer_joint_grpo_live.py",
            "umminfer_online_rollout.py",
            "umminfer_online_grpo_preflight.py",
        }
        for path in (ROOT / "python").glob("*.py"):
            if path.name in offline_dataset_consumers:
                continue
            self.assertNotIn(
                "umminfer_sft_dataset",
                path.read_text(encoding="utf-8", errors="ignore"),
                str(path),
            )
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)
        self.assertNotIn("transformers", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
