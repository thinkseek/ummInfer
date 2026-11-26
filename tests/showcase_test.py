#!/usr/bin/env python3
"""Unit and integration tests for the North-Star Native Multimodal Showcase Suite."""

from __future__ import annotations

import base64
import copy
from http.server import ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import wave


REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
MANIFEST_PATH = REPO_ROOT / "benchmarks" / "showcase" / "prompts.json"
SCRIPT_PATH = TOOLS_DIR / "umminfer_showcase.py"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import umminfer_showcase as showcase  # noqa: E402


class ShowcaseSuiteTest(unittest.TestCase):
    def _sample_stream(self) -> dict:
        png = showcase.MOCK_PNG_J1
        pcm = showcase.MOCK_PCM_J1
        part1 = "At dawn the golden accretion disk glows."
        part2 = " Light curves gracefully around the dark shadow."
        events = [
            showcase._mock_envelope(0, "vlm_delta", part1),
            showcase._mock_envelope(
                1,
                "visual_slot",
                stage="image",
                slot_id="slot-0",
                slot_index=0,
                media_type="image",
                prompt="golden accretion disk",
                text_offset=len(part1),
            ),
            *showcase._mock_artifact_frames(
                2, "art-audio-0", "audio", "audio/pcm", pcm, "tts", stream_index=0
            ),
            showcase._mock_envelope(3, "vlm_delta", part2),
            *showcase._mock_artifact_frames(
                4,
                "art-img-0",
                "image",
                "image/png",
                png,
                "image",
                slot_id="slot-0",
                slot_index=0,
            ),
            showcase._mock_envelope(
                5,
                "complete",
                part1 + part2,
                finish="stop",
                legs=[
                    {"stage": "image", "slot_id": "slot-0", "slot_index": 0, "state": "completed"},
                    {"stage": "tts", "state": "completed"},
                ],
            ),
        ]
        observations = [
            {"time": 0.01 * (i + 1), "event": ev["unified"], "frame": ev}
            for i, ev in enumerate(events)
        ]
        payloads = [json.dumps(ev) for ev in events] + ["[DONE]"]
        return {"observations": observations, "payloads": payloads, "total_s": 0.1}

    def _sample_journey_5_stream(self) -> dict:
        png = showcase.MOCK_PNG_J5
        mp4 = showcase.MOCK_MP4_J5
        pcm = showcase.MOCK_PCM_J5
        part1 = "Above the misty coastal cliffs, a lone wooden lighthouse glows in amber light."
        part2 = " Moments later, the lantern beam sweeps steadily across rolling twilight waves."
        events = [
            showcase._mock_envelope(0, "vlm_delta", part1),
            showcase._mock_envelope(
                1,
                "visual_slot",
                stage="image",
                slot_id="slot-0",
                slot_index=0,
                media_type="image",
                prompt="lone wooden lighthouse glowing above misty coastal cliffs",
                text_offset=len(part1),
                continuity_mode="i2v_chain",
            ),
            *showcase._mock_artifact_frames(
                2, "art-audio-0", "audio", "audio/pcm", pcm, "tts", stream_index=0
            ),
            showcase._mock_envelope(3, "vlm_delta", part2),
            showcase._mock_envelope(
                4,
                "visual_slot",
                stage="video",
                slot_id="slot-1",
                slot_index=1,
                media_type="video",
                prompt="amber lighthouse beam sweeping steadily across rolling waves",
                text_offset=len(part1 + part2),
                continuity_mode="i2v_chain",
                anchor_slot_index=0,
            ),
            *showcase._mock_artifact_frames(
                5,
                "art-img-0",
                "image",
                "image/png",
                png,
                "image",
                slot_id="slot-0",
                slot_index=0,
            ),
            *showcase._mock_artifact_frames(
                6,
                "art-vid-1",
                "video",
                "video/mp4",
                mp4,
                "video",
                slot_id="slot-1",
                slot_index=1,
            ),
            showcase._mock_envelope(
                7,
                "complete",
                part1 + part2,
                finish="stop",
                legs=[
                    {"stage": "image", "slot_id": "slot-0", "slot_index": 0, "state": "completed"},
                    {"stage": "video", "slot_id": "slot-1", "slot_index": 1, "state": "completed"},
                    {"stage": "tts", "state": "completed"},
                ],
            ),
        ]
        observations = [
            {"time": 0.01 * (i + 1), "event": ev["unified"], "frame": ev}
            for i, ev in enumerate(events)
        ]
        payloads = [json.dumps(ev) for ev in events] + ["[DONE]"]
        return {"observations": observations, "payloads": payloads, "total_s": 0.12}

    def test_self_check_cli_passes_and_writes_verified_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="showcase-cli-test-") as tmp:
            out_dir = Path(tmp)
            proc = subprocess.run(
                [sys.executable, "-I", str(SCRIPT_PATH), "--self-check", "--out-dir", str(out_dir)],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"stderr={proc.stderr}\nstdout={proc.stdout}")
            self.assertIn("SELF-CHECK PASS", proc.stdout)
            self.assertIn("OFFLINE_SELF_CHECK", proc.stdout)
            self.assertIn("REAL_MODEL: NOT_RUN", proc.stdout)
            self.assertIn("5/5 journeys passed", proc.stdout)

            report_path = out_dir / "showcase_report.json"
            md_path = out_dir / "showcase_report.md"
            sums_path = out_dir / "SHA256SUMS"
            self.assertTrue(report_path.is_file())
            self.assertTrue(md_path.is_file())
            self.assertTrue(sums_path.is_file())

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["evidence_class"], "OFFLINE_SELF_CHECK")
            self.assertEqual(report["real_model"], "NOT_RUN")
            self.assertTrue(report["summary"]["all_passed"])
            self.assertEqual(report["summary"]["journeys_passed"], 5)
            self.assertEqual(report["summary"]["journeys_total"], 5)

            # Verify Journey 1 artifacts (PNG, MP4, WAV)
            png_bytes = (out_dir / "journey1_slot-0.png").read_bytes()
            self.assertTrue(png_bytes.startswith(showcase.PNG_MAGIC))
            mp4_j1 = (out_dir / "journey1_slot-1.mp4").read_bytes()
            self.assertEqual(mp4_j1[4:8], b"ftyp")
            wav_path = out_dir / "journey1_audio.wav"
            with wave.open(str(wav_path), "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertEqual(wf.getframerate(), 24000)
                self.assertGreater(wf.getnframes(), 0)

            # Verify Journey 2 MP4
            mp4_j2 = (out_dir / "journey2_i2v.mp4").read_bytes()
            self.assertEqual(mp4_j2[4:8], b"ftyp")

            # Verify Journey 3 multi-turn video QA
            j3 = report["journeys"]["journey_3"]
            self.assertEqual(len(j3["turns"]), 3)
            self.assertEqual([t["message_count_sent"] for t in j3["turns"]], [1, 3, 5])
            self.assertGreater(j3["turn2_speedup_vs_turn1"], 1.5)
            self.assertGreater(j3["turn3_speedup_vs_turn1"], 1.5)

            # Verify Journey 4 routing contrast
            j4 = report["journeys"]["journey_4"]
            self.assertEqual(j4["pure_text_visual_slots"], 0)
            self.assertEqual(j4["visual_scene_visual_slots"], 2)
            self.assertTrue(j4["zero_redundant_visuals_on_pure_text"])
            self.assertTrue(j4["visual_routing_active_on_scene"])

            # Verify Journey 5 storyboard_i2v_chain
            j5 = report["journeys"]["journey_5"]
            self.assertEqual(j5["journey_id"], "storyboard_i2v_chain")
            self.assertTrue(j5["i2v_chain_verified"])
            self.assertEqual(j5["slot_1_continuity_mode"], "i2v_chain")
            self.assertEqual(j5["slot_1_anchor_slot_index"], 0)
            self.assertIsNotNone(j5["ttft_s"])
            self.assertIsNotNone(j5["ttfa_s"])
            png_j5 = (out_dir / "journey5_slot-0.png").read_bytes()
            self.assertTrue(png_j5.startswith(showcase.PNG_MAGIC))
            mp4_j5 = (out_dir / "journey5_slot-1.mp4").read_bytes()
            self.assertEqual(mp4_j5[4:8], b"ftyp")
            self.assertEqual(showcase.sha256_bytes(png_j5), j5["slot_0_png_sha256"])
            self.assertEqual(showcase.sha256_bytes(mp4_j5), j5["slot_1_mp4_sha256"])
            wav_j5 = out_dir / "journey5_audio.wav"
            with wave.open(str(wav_j5), "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertEqual(wf.getframerate(), 24000)
                self.assertGreater(wf.getnframes(), 0)

            # Verify SHA256SUMS integrity and Markdown contents
            verified = showcase.verify_sha256sums_file(out_dir, sums_path)
            self.assertGreaterEqual(verified, 11)
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("OFFLINE_SELF_CHECK", md_text)
            self.assertIn("REAL_MODEL: NOT_RUN", md_text)
            for heading in (
                "## Journey 1:",
                "## Journey 2:",
                "## Journey 3:",
                "## Journey 4:",
                "## Journey 5:",
                "## Verified Artifacts",
            ):
                self.assertIn(heading, md_text)

    def test_explicit_anchor_contract_and_hash_tamper(self) -> None:
        linked = self._sample_journey_5_stream()
        source_sha = showcase.sha256_bytes(showcase.MOCK_PNG_J5)
        linked["observations"][4]["event"]["anchor_explicit"] = True
        video = linked["observations"][6]["event"]["artifact"]
        video.update({
            "anchor_slot_index": 0,
            "anchor_explicit": True,
            "anchor_sha256": source_sha,
        })
        for leg in linked["observations"][7]["event"]["legs"]:
            if leg.get("slot_id") == "slot-1":
                leg.update({
                    "anchor_slot_index": 0,
                    "anchor_explicit": True,
                    "anchor_sha256": source_sha,
                })
        metrics, _, _ = showcase.validate_narrative_stream(
            linked, max_slots=2, require_audio=True)
        self.assertEqual(metrics["slots"][1]["anchor_sha256"], source_sha)
        self.assertTrue(metrics["slots"][1]["anchor_explicit"])
        showcase.validate_journey_5_stream(linked, max_slots=2)

        bad = copy.deepcopy(linked)
        bad["observations"][6]["event"]["artifact"]["anchor_sha256"] = "0" * 64
        bad["observations"][7]["event"]["legs"][1]["anchor_sha256"] = "0" * 64
        with self.assertRaisesRegex(AssertionError, "anchor SHA-256"):
            showcase.validate_narrative_stream(bad, max_slots=2, require_audio=True)

    def test_journey_selection_flag_supports_5_and_all(self) -> None:
        self.assertEqual(showcase.parse_journeys("all"), [1, 2, 3, 4, 5])
        self.assertEqual(showcase.parse_journeys("5"), [5])
        with tempfile.TemporaryDirectory(prefix="showcase-j5-only-") as tmp:
            out_dir = Path(tmp)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(SCRIPT_PATH),
                    "--self-check",
                    "--journey",
                    "5",
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"stderr={proc.stderr}\nstdout={proc.stdout}")
            self.assertIn("1/1 journeys passed", proc.stdout)
            report = json.loads((out_dir / "showcase_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["journeys_requested"], [5])
            self.assertEqual(list(report["journeys"].keys()), ["journey_5"])
            self.assertTrue(report["journeys"]["journey_5"]["i2v_chain_verified"])

    def test_journey_5_negative_validation_fails_closed(self) -> None:
        baseline = self._sample_journey_5_stream()
        metrics, _, _ = showcase.validate_journey_5_stream(baseline, max_slots=2)
        self.assertTrue(metrics["i2v_chain_verified"])
        self.assertEqual(metrics["slot_1_anchor_slot_index"], 0)
        self.assertEqual(metrics["slot_1_continuity_mode"], "i2v_chain")

        # 1. Wrong anchor_slot_index (!= 0) must fail closed
        bad_anchor = copy.deepcopy(baseline)
        bad_anchor["observations"][4]["event"]["anchor_slot_index"] = 1
        with self.assertRaises(AssertionError) as ctx:
            showcase.validate_journey_5_stream(bad_anchor, max_slots=2)
        self.assertIn("anchor_slot_index", str(ctx.exception))

        # 2. Missing anchor_slot_index must fail closed
        missing_anchor = copy.deepcopy(baseline)
        del missing_anchor["observations"][4]["event"]["anchor_slot_index"]
        with self.assertRaises(AssertionError) as ctx:
            showcase.validate_journey_5_stream(missing_anchor, max_slots=2)
        self.assertIn("anchor_slot_index", str(ctx.exception))

        # 3. Wrong continuity_mode on Slot 1 must fail closed
        bad_mode = copy.deepcopy(baseline)
        bad_mode["observations"][4]["event"]["continuity_mode"] = "prompt_anchor"
        with self.assertRaises(AssertionError) as ctx:
            showcase.validate_journey_5_stream(bad_mode, max_slots=2)
        self.assertIn("continuity_mode", str(ctx.exception))

        # 4. Missing video artifact for Slot 1 must fail closed
        missing_video = copy.deepcopy(baseline)
        missing_video["observations"] = [
            obs for obs in missing_video["observations"]
            if not (obs["event"].get("kind") == "artifact" and obs["event"].get("stage") == "video")
        ]
        for idx, obs in enumerate(missing_video["observations"]):
            obs["event"]["seq"] = idx
        with self.assertRaises(AssertionError) as ctx:
            showcase.validate_journey_5_stream(missing_video, max_slots=2)
        self.assertIn("artifact", str(ctx.exception))

    def test_tampered_artifact_sha256_fails_closed(self) -> None:
        stream = self._sample_stream()
        # Baseline must pass
        showcase.validate_narrative_stream(stream, require_audio=True, require_mid_stream_slot=True)

        tampered = copy.deepcopy(stream)
        for obs in tampered["observations"]:
            if obs["event"].get("kind") == "artifact" and obs["event"].get("stage") == "image":
                obs["event"]["artifact"]["sha256"] = "0" * 64
        with self.assertRaises(AssertionError) as ctx:
            showcase.validate_narrative_stream(
                tampered, require_audio=True, require_mid_stream_slot=True
            )
        self.assertIn("SHA-256", str(ctx.exception))

    def test_tampered_artifact_bytes_fails_closed(self) -> None:
        tampered = self._sample_stream()
        for obs in tampered["observations"]:
            if obs["event"].get("kind") == "artifact" and obs["event"].get("stage") == "image":
                obs["event"]["artifact"]["bytes"] = 999999
        with self.assertRaises(AssertionError) as ctx:
            showcase.validate_narrative_stream(
                tampered, require_audio=True, require_mid_stream_slot=True
            )
        self.assertIn("byte count", str(ctx.exception))

    def test_leaked_visual_marker_in_prose_fails_closed(self) -> None:
        leaked = self._sample_stream()
        bad_text = "At dawn [[image: glowing red lantern]] the disk glows."
        leaked["observations"][0]["event"]["text"] = bad_text
        leaked["observations"][0]["frame"]["choices"][0]["delta"]["content"] = bad_text
        leaked["observations"][-1]["event"]["text"] = (
            bad_text + leaked["observations"][3]["event"]["text"]
        )
        with self.assertRaises(AssertionError) as ctx:
            showcase.validate_narrative_stream(
                leaked, require_audio=True, require_mid_stream_slot=True
            )
        self.assertIn("clean prose", str(ctx.exception))

    def test_non_monotonic_sse_seq_fails_closed(self) -> None:
        broken = self._sample_stream()
        broken["observations"][3]["event"]["seq"] = 99
        with self.assertRaises(AssertionError) as ctx:
            showcase.validate_narrative_stream(
                broken, require_audio=True, require_mid_stream_slot=True
            )
        self.assertIn("monotonic", str(ctx.exception))

    def test_url_transport_hygiene(self) -> None:
        self.assertEqual(
            showcase.validate_server_url("http://127.0.0.1:43301/"),
            "http://127.0.0.1:43301",
        )
        self.assertEqual(
            showcase.validate_server_url("http://localhost:43301"),
            "http://localhost:43301",
        )
        self.assertEqual(
            showcase.validate_server_url("http://[::1]:43301"),
            "http://[::1]:43301",
        )
        self.assertEqual(
            showcase.validate_server_url("https://api.example.com:8443"),
            "https://api.example.com:8443",
        )
        with self.assertRaises(ValueError):
            showcase.validate_server_url("http://remote.example.com:43301")
        self.assertEqual(
            showcase.validate_server_url(
                "http://remote.example.com:43301", allow_remote_http=True
            ),
            "http://remote.example.com:43301",
        )
        with self.assertRaises(ValueError):
            showcase.validate_server_url("http://user:secret@127.0.0.1:43301")

    def test_prompts_manifest_valid_and_free_of_internal_tokens(self) -> None:
        raw_text = MANIFEST_PATH.read_text(encoding="utf-8")
        data = showcase.load_manifest(MANIFEST_PATH)
        journeys = data["journeys"]

        # Verify bounded token limits and fixed seeds on all journeys
        j1 = journeys["journey_1_interleaved_narrative"]
        self.assertLessEqual(j1["max_tokens"], 512)
        self.assertEqual(j1["seed"], 42)
        self.assertEqual(j1["narrative"], {"enabled": True, "visual_mode": "auto", "max_slots": 2})
        self.assertEqual(j1["audio"]["voice"], "Chelsie")
        self.assertTrue(j1["audio"]["stream"])
        self.assertEqual(j1["audio"]["format"], "wav")

        j2 = journeys["journey_2_i2v_animation"]
        self.assertEqual(j2["kind"], "video")
        self.assertEqual(j2["size"], "704x480")
        self.assertEqual(j2["num_frames"], 29)
        self.assertEqual(j2["steps"], 15)
        self.assertEqual(j2["seed"], 42)

        j3 = journeys["journey_3_multiturn_video_qa"]
        self.assertEqual(j3["temperature"], 0.0)
        self.assertEqual(len(j3["turns"]), 3)
        self.assertLessEqual(j3["max_tokens"], 256)

        j4 = journeys["journey_4_modality_routing_contrast"]
        self.assertEqual(
            j4["shared_options"]["narrative"],
            {
                "enabled": True,
                "visual_mode": "auto",
                "max_slots": 2,
                "allow_text_only": True,
            },
        )
        self.assertEqual(j4["pure_text_case"]["expected_visual_slots"], 0)
        self.assertGreaterEqual(j4["visual_scene_case"]["expected_min_visual_slots"], 1)

        j5 = journeys["journey_5"]
        self.assertEqual(j5["id"], "storyboard_i2v_chain")
        self.assertEqual(j5["modalities"], ["text", "audio", "image", "video"])
        self.assertEqual(j5["narrative"]["enabled"], True)
        self.assertEqual(j5["narrative"]["visual_mode"], "auto")
        self.assertEqual(j5["narrative"]["max_slots"], 2)
        self.assertEqual(j5["narrative"]["continuity_mode"], "i2v_chain")
        self.assertEqual(
            j5["narrative"]["style_anchor"],
            "cinematic warm golden glow, misty coastal twilight, 35mm film",
        )
        self.assertEqual(j5["video"]["size"], "704x480")
        self.assertEqual(j5["video"]["num_frames"], 29)
        self.assertEqual(j5["video"]["steps"], 15)
        self.assertEqual(j5["video"]["teacache_thresh"], 0.05)
        self.assertEqual(j5["audio"]["voice"], "vivian")
        self.assertEqual(j5["audio"]["format"], "wav")
        self.assertTrue(j5["audio"]["stream"])
        self.assertIn("[[image:", j5["prompt"])
        self.assertIn("[[video:", j5["prompt"])

        # Dynamically constructed forbidden internal tokens (never literal in test source)
        forbidden = [
            "h" + "20b",
            "/" + "Users/",
            "/" + "home/",
            "byte" + "dance",
            "10." + "0.",
            "192." + "168.",
            "172." + "16.",
            "http:" + "//",
            "https:" + "//",
        ]
        lowered = raw_text.lower()
        for token in forbidden:
            self.assertNotIn(
                token.lower(),
                lowered,
                f"Forbidden internal/host token found in {MANIFEST_PATH.name}",
            )

    def test_report_markdown_and_sha256sums_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="showcase-det-test-") as tmp:
            out_dir = Path(tmp)
            report = showcase.run_self_check(MANIFEST_PATH, out_dir=out_dir)

            md_a = showcase.render_markdown_report(report)
            md_b = showcase.render_markdown_report(report)
            self.assertEqual(md_a, md_b)

            tracked = [art["filename"] for art in report["artifacts"]] + [
                "showcase_report.json",
                "showcase_report.md",
            ]
            sums_a = showcase.build_sha256sums(out_dir, tracked)
            sums_b = showcase.build_sha256sums(out_dir, list(reversed(tracked)))
            self.assertEqual(sums_a, sums_b)

            # Check sorted order in SHA256SUMS
            listed_names = [line.split("  ", 1)[1] for line in sums_a.strip().splitlines()]
            self.assertEqual(listed_names, sorted(listed_names))

            # Default OFF must not inject telemetry or behavior keys or sections
            self.assertNotIn("native_allocator_telemetry", report)
            self.assertNotIn("behavior_logprobs", report)
            for jkey in ("journey_1", "journey_2", "journey_3", "journey_4", "journey_5"):
                self.assertNotIn("native_allocator_telemetry", report["journeys"][jkey])
                self.assertNotIn("behavior_logprobs", report["journeys"][jkey])
            self.assertNotIn("## Native Allocator Telemetry", md_a)
            self.assertNotIn("## Behavior Logprobs Evidence", md_a)
            self.assertNotIn("Behavior:", md_a)

    def test_native_allocator_telemetry_opt_in_cli_covered(self) -> None:
        with tempfile.TemporaryDirectory(prefix="showcase-tel-covered-") as tmp:
            out_dir = Path(tmp)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(SCRIPT_PATH),
                    "--self-check",
                    "--native-allocator-telemetry",
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"stderr={proc.stderr}\nstdout={proc.stdout}")
            self.assertIn("SELF-CHECK PASS", proc.stdout)

            report = json.loads((out_dir / "showcase_report.json").read_text(encoding="utf-8"))
            md_text = (out_dir / "showcase_report.md").read_text(encoding="utf-8")
            verified = showcase.verify_sha256sums_file(out_dir, out_dir / "SHA256SUMS")
            self.assertGreaterEqual(verified, 11)

            top_tel = report["native_allocator_telemetry"]
            self.assertTrue(top_tel["enabled"])
            self.assertEqual(top_tel["scope"], "request-native-allocator")
            self.assertEqual(top_tel["coverage"], "COVERED")
            self.assertEqual(top_tel["covered_stages"], top_tel["total_stages"])
            self.assertGreater(top_tel["total_stages"], 0)
            self.assertEqual(top_tel["max_allocated_bytes_peak"], 335544320)
            self.assertEqual(top_tel["max_reserved_bytes_peak"], 335544320)
            self.assertEqual(top_tel["reasons"], [])
            self.assertEqual(top_tel["whole_device_nvml"], "separate_measurement_resources")

            # Journey 1, 3, 4, 5 carry validated native_allocator_telemetry; Journey 2 does not.
            j1_tel = report["journeys"]["journey_1"]["native_allocator_telemetry"]
            self.assertEqual(j1_tel["coverage"], "COVERED")
            self.assertEqual(j1_tel["covered_stages"], 4)
            self.assertEqual(j1_tel["total_stages"], 4)
            self.assertEqual(j1_tel["max_allocated_bytes_peak"], 301989888)
            self.assertEqual(j1_tel["whole_device_nvml"], "separate_measurement_resources")

            self.assertNotIn("native_allocator_telemetry", report["journeys"]["journey_2"])

            j3 = report["journeys"]["journey_3"]
            self.assertEqual(j3["native_allocator_telemetry"]["coverage"], "COVERED")
            self.assertEqual(j3["native_allocator_telemetry"]["requests"], 3)
            self.assertEqual(j3["native_allocator_telemetry"]["max_allocated_bytes_peak"], 234881024)
            for turn in j3["turns"]:
                self.assertEqual(turn["native_allocator_telemetry"]["coverage"], "COVERED")
                self.assertEqual(
                    turn["native_allocator_telemetry"]["request_id"],
                    f"showcase-video-t{turn['turn']}",
                )

            j4 = report["journeys"]["journey_4"]
            self.assertEqual(j4["pure_text"]["native_allocator_telemetry"]["max_allocated_bytes_peak"], 134217728)
            self.assertEqual(j4["visual_scene"]["native_allocator_telemetry"]["max_allocated_bytes_peak"], 301989888)
            self.assertEqual(j4["native_allocator_telemetry"]["max_allocated_bytes_peak"], 301989888)

            j5_tel = report["journeys"]["journey_5"]["native_allocator_telemetry"]
            self.assertEqual(j5_tel["coverage"], "COVERED")
            self.assertEqual(j5_tel["max_allocated_bytes_peak"], 335544320)

            self.assertIn("## Native Allocator Telemetry", md_text)
            self.assertIn("320.0 MiB peak", md_text)
            self.assertIn("288.0 MiB (301989888 B)", md_text)
            self.assertIn("128.0 MiB (134217728 B)", md_text)
            self.assertIn("separate_measurement_resources", md_text)

    def test_native_allocator_telemetry_honest_not_covered_and_partial(self) -> None:
        def _not_covered_factory(_req, request_id, stage_specs):
            stages = [
                showcase._mock_telemetry_stage(
                    stage_name,
                    request_id=request_id,
                    coverage="NOT_COVERED",
                    reason="batched_request_attribution_unavailable",
                )
                for stage_name, _ in stage_specs
            ]
            return showcase._mock_telemetry_envelope(request_id, stages)

        with tempfile.TemporaryDirectory(prefix="showcase-tel-nc-") as tmp:
            out_dir = Path(tmp)
            report = showcase.run_self_check(
                MANIFEST_PATH,
                out_dir=out_dir,
                native_allocator_telemetry=True,
                telemetry_factory=_not_covered_factory,
            )
            top_tel = report["native_allocator_telemetry"]
            self.assertEqual(top_tel["coverage"], "NOT_COVERED")
            self.assertEqual(top_tel["covered_stages"], 0)
            self.assertGreater(top_tel["total_stages"], 0)
            self.assertIsNone(top_tel["max_allocated_bytes_peak"])
            self.assertIsNone(top_tel["max_reserved_bytes_peak"])
            self.assertEqual(top_tel["reasons"], ["batched_request_attribution_unavailable"])
            for jkey in ("journey_1", "journey_3", "journey_4", "journey_5"):
                jtel = report["journeys"][jkey]["native_allocator_telemetry"]
                self.assertEqual(jtel["coverage"], "NOT_COVERED")
                self.assertIsNone(jtel["max_allocated_bytes_peak"])
                self.assertIsNone(jtel["max_reserved_bytes_peak"])
                for stage in jtel["stages"]:
                    self.assertIsNone(stage["allocated_bytes_peak"])
                    self.assertIsNone(stage["reserved_bytes_peak"])

            md_text = (out_dir / "showcase_report.md").read_text(encoding="utf-8")
            self.assertIn("## Native Allocator Telemetry", md_text)
            self.assertIn("N/C (batched_request_attribution_unavailable)", md_text)
            self.assertNotIn("0 B", md_text)
            self.assertNotIn("0.0 MiB", md_text)

        # Partial coverage: vlm COVERED + tts NOT_COVERED
        partial_env = showcase._mock_telemetry_envelope(
            "showcase-mock-1",
            [
                showcase._mock_telemetry_stage(
                    "vlm", request_id="showcase-mock-1", coverage="COVERED", peak_bytes=268435456
                ),
                showcase._mock_telemetry_stage(
                    "tts",
                    request_id="showcase-mock-1",
                    coverage="NOT_COVERED",
                    reason="backend_not_instrumented",
                ),
            ],
        )
        summary = showcase.validate_native_allocator_telemetry(
            partial_env, expected_request_id="showcase-mock-1"
        )
        self.assertEqual(summary["coverage"], "PARTIAL")
        self.assertEqual(summary["covered_stages"], 1)
        self.assertEqual(summary["total_stages"], 2)
        self.assertEqual(summary["max_allocated_bytes_peak"], 268435456)
        self.assertEqual(summary["reasons"], ["backend_not_instrumented"])
        self.assertEqual(
            showcase.format_native_allocator_summary(summary),
            "256.0 MiB peak (covered=1/2) [N/C: backend_not_instrumented]",
        )

    def test_native_allocator_telemetry_fails_closed_on_malformed_envelopes(self) -> None:
        base_stream = self._sample_stream()
        valid_env = showcase._mock_telemetry_envelope(
            "showcase-mock-1",
            [
                showcase._mock_telemetry_stage(
                    "vlm", request_id="showcase-mock-1", coverage="COVERED", peak_bytes=134217728
                )
            ],
        )

        # 1. Missing telemetry when required fails closed
        with self.assertRaisesRegex(AssertionError, "telemetry"):
            showcase.validate_narrative_stream(
                base_stream,
                require_audio=True,
                require_mid_stream_slot=True,
                require_native_allocator_telemetry=True,
            )

        # Baseline valid telemetry passes
        ok_stream = copy.deepcopy(base_stream)
        ok_stream["observations"][-1]["event"]["telemetry"] = copy.deepcopy(valid_env)
        metrics, _, _ = showcase.validate_narrative_stream(
            ok_stream,
            require_audio=True,
            require_mid_stream_slot=True,
            require_native_allocator_telemetry=True,
        )
        self.assertEqual(metrics["native_allocator_telemetry"]["max_allocated_bytes_peak"], 134217728)

        # 2. Boolean peak rejected (allocated_bytes_peak=True / reserved_bytes_peak=False)
        for field, bad_val in (
            ("allocated_bytes_peak", True),
            ("allocated_bytes_peak", False),
            ("reserved_bytes_peak", True),
        ):
            bad_env = copy.deepcopy(valid_env)
            bad_env["stages"][0][field] = bad_val
            with self.assertRaisesRegex(AssertionError, "not bool"):
                showcase.validate_native_allocator_telemetry(bad_env, expected_request_id="showcase-mock-1")

        # 3. Negative peak rejected
        for field in ("allocated_bytes_peak", "reserved_bytes_peak"):
            bad_env = copy.deepcopy(valid_env)
            bad_env["stages"][0][field] = -1
            with self.assertRaisesRegex(AssertionError, "non-negative integer"):
                showcase.validate_native_allocator_telemetry(bad_env, expected_request_id="showcase-mock-1")

        # 4. request_id mismatch rejected on stream and envelope
        mismatch_stream = copy.deepcopy(ok_stream)
        mismatch_stream["observations"][-1]["event"]["telemetry"]["request_id"] = "other-req-999"
        with self.assertRaisesRegex(AssertionError, "request_id mismatch"):
            showcase.validate_narrative_stream(
                mismatch_stream,
                require_audio=True,
                require_mid_stream_slot=True,
                require_native_allocator_telemetry=True,
            )

        # 5. NOT_COVERED with fabricated 0 peak rejected (never coerce None -> 0)
        nc_zero = showcase._mock_telemetry_envelope(
            "showcase-mock-1",
            [
                showcase._mock_telemetry_stage(
                    "vlm", request_id="showcase-mock-1", coverage="NOT_COVERED", reason="backend_not_instrumented"
                )
            ],
        )
        nc_zero["stages"][0]["allocated_bytes_peak"] = 0
        with self.assertRaisesRegex(AssertionError, "null peaks"):
            showcase.validate_native_allocator_telemetry(nc_zero, expected_request_id="showcase-mock-1")

        # 6. COVERED with null peak rejected
        cov_null = copy.deepcopy(valid_env)
        cov_null["stages"][0]["allocated_bytes_peak"] = None
        with self.assertRaisesRegex(AssertionError, "non-negative integer"):
            showcase.validate_native_allocator_telemetry(cov_null, expected_request_id="showcase-mock-1")

        # 7. Boolean schema_version, empty stages, wrong scope, invalid coverage, non-string NVML marker
        bad_schema = copy.deepcopy(valid_env)
        bad_schema["schema_version"] = True
        with self.assertRaisesRegex(AssertionError, "schema_version"):
            showcase.validate_native_allocator_telemetry(bad_schema)

        empty_stages = copy.deepcopy(valid_env)
        empty_stages["stages"] = []
        with self.assertRaisesRegex(AssertionError, "non-empty list"):
            showcase.validate_native_allocator_telemetry(empty_stages)

        bad_scope = copy.deepcopy(valid_env)
        bad_scope["scope"] = "whole-device-nvml"
        with self.assertRaisesRegex(AssertionError, "scope"):
            showcase.validate_native_allocator_telemetry(bad_scope)

        bad_nvml = copy.deepcopy(valid_env)
        bad_nvml["whole_device_nvml"] = {"used_bytes": 12345}
        with self.assertRaisesRegex(AssertionError, "whole_device_nvml"):
            showcase.validate_native_allocator_telemetry(bad_nvml)

        # 8. End-to-end loopback fail-closed when server omits telemetry on opt-in request
        with self.assertRaisesRegex(AssertionError, "telemetry"):
            showcase.run_self_check(
                MANIFEST_PATH,
                journeys=[1],
                native_allocator_telemetry=True,
                telemetry_factory=lambda _req, _rid, _specs: None,
            )

        # 9. End-to-end loopback fail-closed on Journey 3 non-stream request_id mismatch
        def _j3_mismatch_factory(req, request_id, stage_specs):
            effective_id = request_id + "-wrong" if not req.get("stream") else request_id
            stages = [
                showcase._mock_telemetry_stage(
                    stage_name, request_id=effective_id, coverage="COVERED", peak_bytes=peak
                )
                for stage_name, peak in stage_specs
            ]
            return showcase._mock_telemetry_envelope(effective_id, stages)

        with self.assertRaisesRegex(AssertionError, "request_id mismatch"):
            showcase.run_self_check(
                MANIFEST_PATH,
                journeys=[1, 2, 3],
                native_allocator_telemetry=True,
                telemetry_factory=_j3_mismatch_factory,
            )

    def _start_mock_server(self) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
        server = ThreadingHTTPServer(("127.0.0.1", 0), showcase._MockShowcaseHandler)
        server.chat_requests = []
        server.render_requests = []
        server.telemetry_factory = None
        server.behavior_factory = None
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread, f"http://127.0.0.1:{server.server_port}"

    def _stop_mock_server(self, server: ThreadingHTTPServer, thread: threading.Thread) -> None:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    def test_video_frame_selection_default_off_omits_field_and_preserves_report(self) -> None:
        manifest = showcase.load_manifest(MANIFEST_PATH)
        j3_spec = manifest["journeys"]["journey_3_multiturn_video_qa"]
        server, thread, base_url = self._start_mock_server()
        try:
            j3 = showcase.run_journey_3(
                base_url,
                "qwen3-native",
                j3_spec,
                video_bytes=showcase.MOCK_MP4_J2,
                timeout=15.0,
            )
            self.assertEqual(len(server.chat_requests), 3)
            expected_url = (
                "data:video/mp4;base64,"
                + base64.b64encode(showcase.MOCK_MP4_J2).decode("ascii")
            )
            for req in server.chat_requests:
                video_part = req["messages"][0]["content"][0]
                self.assertEqual(video_part["type"], "video_url")
                self.assertEqual(video_part["video_url"], {"url": expected_url})
                self.assertNotIn("frame_selection", video_part["video_url"])
            self.assertNotIn("video_frame_selection", j3)
        finally:
            self._stop_mock_server(server, thread)

        with tempfile.TemporaryDirectory(prefix="showcase-j3-default-") as tmp:
            out_dir = Path(tmp)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(SCRIPT_PATH),
                    "--mock",
                    "--journey",
                    "3",
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"stderr={proc.stderr}\nstdout={proc.stdout}")
            self.assertIn("1/1 journeys passed", proc.stdout)
            report = json.loads((out_dir / "showcase_report.json").read_text(encoding="utf-8"))
            md_text = (out_dir / "showcase_report.md").read_text(encoding="utf-8")
            self.assertNotIn("video_frame_selection", report["journeys"]["journey_3"])
            self.assertNotIn("Video Frame Selection", md_text)
            self.assertNotIn("frame_selection=", md_text)
            self.assertEqual(
                showcase.verify_sha256sums_file(out_dir, out_dir / "SHA256SUMS"),
                2,
            )

    def test_video_frame_selection_uniform_opt_in(self) -> None:
        manifest = showcase.load_manifest(MANIFEST_PATH)
        j3_spec = manifest["journeys"]["journey_3_multiturn_video_qa"]
        server, thread, base_url = self._start_mock_server()
        try:
            j3 = showcase.run_journey_3(
                base_url,
                "qwen3-native",
                j3_spec,
                video_bytes=showcase.MOCK_MP4_J2,
                timeout=15.0,
                video_frame_selection={"policy": "uniform"},
            )
            self.assertEqual(len(server.chat_requests), 3)
            turn1_video_url = server.chat_requests[0]["messages"][0]["content"][0]["video_url"]
            self.assertEqual(turn1_video_url["frame_selection"], {"policy": "uniform"})
            self.assertEqual(j3["video_frame_selection"], {"policy": "uniform"})
        finally:
            self._stop_mock_server(server, thread)

        with tempfile.TemporaryDirectory(prefix="showcase-j3-uniform-") as tmp:
            out_dir = Path(tmp)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(SCRIPT_PATH),
                    "--mock",
                    "--journey",
                    "3",
                    "--video-frame-policy",
                    "uniform",
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"stderr={proc.stderr}\nstdout={proc.stdout}")
            self.assertIn("1/1 journeys passed", proc.stdout)
            report = json.loads((out_dir / "showcase_report.json").read_text(encoding="utf-8"))
            md_text = (out_dir / "showcase_report.md").read_text(encoding="utf-8")
            self.assertEqual(
                report["journeys"]["journey_3"]["video_frame_selection"],
                {"policy": "uniform"},
            )
            self.assertIn("frame_selection=policy=uniform", md_text)
            self.assertIn("- **Video Frame Selection**: `policy=uniform`", md_text)
            self.assertNotIn("motion_threshold", md_text)
            self.assertNotIn("scene_threshold", md_text)
            self.assertEqual(
                showcase.verify_sha256sums_file(out_dir, out_dir / "SHA256SUMS"),
                2,
            )

    def test_video_frame_selection_temporal_opt_in(self) -> None:
        manifest = showcase.load_manifest(MANIFEST_PATH)
        j3_spec = manifest["journeys"]["journey_3_multiturn_video_qa"]

        # Exact thresholds sent on Turn 1; single-threshold omits null keys.
        server, thread, base_url = self._start_mock_server()
        try:
            j3_both = showcase.run_journey_3(
                base_url,
                "qwen3-native",
                j3_spec,
                video_bytes=showcase.MOCK_MP4_J2,
                timeout=15.0,
                video_frame_selection={
                    "policy": "temporal",
                    "motion_threshold": 0.25,
                    "scene_threshold": 0.4,
                },
            )
            self.assertEqual(
                server.chat_requests[0]["messages"][0]["content"][0]["video_url"]["frame_selection"],
                {"policy": "temporal", "motion_threshold": 0.25, "scene_threshold": 0.4},
            )
            self.assertEqual(
                j3_both["video_frame_selection"],
                {"policy": "temporal", "motion_threshold": 0.25, "scene_threshold": 0.4},
            )

            server.chat_requests.clear()
            j3_motion_only = showcase.run_journey_3(
                base_url,
                "qwen3-native",
                j3_spec,
                video_bytes=showcase.MOCK_MP4_J2,
                timeout=15.0,
                video_frame_selection={"policy": "temporal", "motion_threshold": 0.0},
            )
            sent_sel = server.chat_requests[0]["messages"][0]["content"][0]["video_url"]["frame_selection"]
            self.assertEqual(sent_sel, {"policy": "temporal", "motion_threshold": 0.0})
            self.assertNotIn("scene_threshold", sent_sel)
            self.assertEqual(
                j3_motion_only["video_frame_selection"],
                {"policy": "temporal", "motion_threshold": 0.0},
            )
        finally:
            self._stop_mock_server(server, thread)

        # CLI E2E with --mock --journey 3 and both thresholds
        with tempfile.TemporaryDirectory(prefix="showcase-j3-temporal-") as tmp:
            out_dir = Path(tmp)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(SCRIPT_PATH),
                    "--mock",
                    "--journey",
                    "3",
                    "--video-frame-policy",
                    "temporal",
                    "--video-motion-threshold",
                    "0.25",
                    "--video-scene-threshold",
                    "0.4",
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"stderr={proc.stderr}\nstdout={proc.stdout}")
            self.assertIn("1/1 journeys passed", proc.stdout)
            report = json.loads((out_dir / "showcase_report.json").read_text(encoding="utf-8"))
            md_text = (out_dir / "showcase_report.md").read_text(encoding="utf-8")
            self.assertEqual(
                report["journeys"]["journey_3"]["video_frame_selection"],
                {"policy": "temporal", "motion_threshold": 0.25, "scene_threshold": 0.4},
            )
            self.assertIn(
                "frame_selection=policy=temporal, motion_threshold=0.25, scene_threshold=0.4",
                md_text,
            )
            self.assertIn(
                "- **Video Frame Selection**: `policy=temporal, motion_threshold=0.25, scene_threshold=0.4`",
                md_text,
            )
            self.assertEqual(
                showcase.verify_sha256sums_file(out_dir, out_dir / "SHA256SUMS"),
                2,
            )

        # CLI E2E with --mock --journey all and single threshold (boundary 1.0)
        with tempfile.TemporaryDirectory(prefix="showcase-all-temporal-") as tmp:
            out_dir = Path(tmp)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(SCRIPT_PATH),
                    "--mock",
                    "--journey",
                    "all",
                    "--video-frame-policy",
                    "temporal",
                    "--video-scene-threshold",
                    "1.0",
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"stderr={proc.stderr}\nstdout={proc.stdout}")
            self.assertIn("5/5 journeys passed", proc.stdout)
            report = json.loads((out_dir / "showcase_report.json").read_text(encoding="utf-8"))
            self.assertEqual(
                report["journeys"]["journey_3"]["video_frame_selection"],
                {"policy": "temporal", "scene_threshold": 1.0},
            )

    def test_video_frame_selection_cli_rejects_invalid_flags_fail_closed(self) -> None:
        # Unit checks on build_video_frame_selection
        self.assertIsNone(showcase.build_video_frame_selection(None))
        self.assertEqual(
            showcase.build_video_frame_selection("uniform"),
            {"policy": "uniform"},
        )
        self.assertEqual(
            showcase.build_video_frame_selection("temporal"),
            {"policy": "temporal"},
        )
        self.assertEqual(
            showcase.build_video_frame_selection(
                "temporal", motion_threshold=0.0, scene_threshold=1.0
            ),
            {"policy": "temporal", "motion_threshold": 0.0, "scene_threshold": 1.0},
        )
        for bad_kwargs in (
            {"policy": None, "motion_threshold": 0.2},
            {"policy": None, "scene_threshold": 0.3},
            {"policy": "uniform", "motion_threshold": 0.2},
            {"policy": "uniform", "scene_threshold": 0.3},
            {"policy": "temporal", "motion_threshold": -0.01},
            {"policy": "temporal", "motion_threshold": 1.01},
            {"policy": "temporal", "scene_threshold": math.nan},
            {"policy": "temporal", "scene_threshold": math.inf},
            {"policy": "temporal", "motion_threshold": -math.inf},
            {"policy": "temporal", "motion_threshold": True},
            {"policy": "adaptive"},
        ):
            with self.assertRaises(ValueError, msg=f"expected ValueError for {bad_kwargs}"):
                showcase.build_video_frame_selection(**bad_kwargs)

        # CLI fail-closed checks
        cli_cases = [
            (["--mock", "--journey", "3", "--video-motion-threshold", "0.2"], "temporal"),
            (["--mock", "--journey", "3", "--video-scene-threshold", "0.35"], "temporal"),
            (
                [
                    "--mock",
                    "--journey",
                    "3",
                    "--video-frame-policy",
                    "uniform",
                    "--video-motion-threshold",
                    "0.2",
                ],
                "temporal",
            ),
            (
                [
                    "--mock",
                    "--journey",
                    "3",
                    "--video-frame-policy",
                    "uniform",
                    "--video-scene-threshold",
                    "0.35",
                ],
                "temporal",
            ),
            (
                [
                    "--mock",
                    "--journey",
                    "3",
                    "--video-frame-policy",
                    "temporal",
                    "--video-motion-threshold",
                    "-0.1",
                ],
                "[0.0, 1.0]",
            ),
            (
                [
                    "--mock",
                    "--journey",
                    "3",
                    "--video-frame-policy",
                    "temporal",
                    "--video-scene-threshold",
                    "1.5",
                ],
                "[0.0, 1.0]",
            ),
            (
                [
                    "--mock",
                    "--journey",
                    "3",
                    "--video-frame-policy",
                    "temporal",
                    "--video-motion-threshold",
                    "nan",
                ],
                "[0.0, 1.0]",
            ),
            (
                [
                    "--mock",
                    "--journey",
                    "3",
                    "--video-frame-policy",
                    "temporal",
                    "--video-scene-threshold",
                    "inf",
                ],
                "[0.0, 1.0]",
            ),
            (
                ["--mock", "--journey", "3", "--video-frame-policy", "adaptive"],
                "invalid choice",
            ),
        ]
        for extra_args, expected_substr in cli_cases:
            proc = subprocess.run(
                [sys.executable, "-I", str(SCRIPT_PATH), *extra_args],
                capture_output=True,
                text=True,
                timeout=15.0,
                check=False,
            )
            self.assertNotEqual(
                proc.returncode,
                0,
                f"expected non-zero exit for {extra_args}; stdout={proc.stdout}",
            )
            self.assertIn(
                expected_substr,
                proc.stderr,
                f"missing {expected_substr!r} in stderr for {extra_args}: {proc.stderr}",
            )

    def test_mock_server_rejects_malformed_video_frame_selection_http_400(self) -> None:
        video_data_url = (
            "data:video/mp4;base64,"
            + base64.b64encode(showcase.MOCK_MP4_J2).decode("ascii")
        )

        def _chat_body(frame_selection_value, include_field: bool = True) -> dict:
            video_url: dict = {"url": video_data_url}
            if include_field:
                video_url["frame_selection"] = frame_selection_value
            return {
                "model": "qwen3-native",
                "stream": False,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": video_url},
                        {"type": "text", "text": "Describe the clip."},
                    ],
                }],
            }

        server, thread, base_url = self._start_mock_server()
        try:
            endpoint = base_url + "/v1/chat/completions"

            # Valid requests return HTTP 200 (default OFF, uniform, temporal, temporal + thresholds)
            for valid_body in (
                _chat_body(None, include_field=False),
                _chat_body({"policy": "uniform"}),
                _chat_body({"policy": "temporal"}),
                _chat_body({"policy": "temporal", "motion_threshold": 0.2, "scene_threshold": 0.35}),
            ):
                status, _, _ = showcase.http_request(endpoint, valid_body, timeout=10.0)
                self.assertEqual(status, 200, f"expected 200 for valid body {valid_body}")

            # Malformed frame_selection payloads return HTTP 400
            bad_selections = [
                "temporal",
                123,
                None,
                [],
                {},
                {"policy": "adaptive"},
                {"policy": "temporal", "max_frames": 16},
                {"policy": "uniform", "motion_threshold": 0.2},
                {"policy": "uniform", "scene_threshold": 0.3},
                {"policy": "temporal", "motion_threshold": -0.01},
                {"policy": "temporal", "scene_threshold": 1.01},
                {"policy": "temporal", "motion_threshold": "0.2"},
                {"policy": "temporal", "motion_threshold": True},
                {"policy": "temporal", "scene_threshold": None},
            ]
            for bad in bad_selections:
                status, _, raw = showcase.http_request(endpoint, _chat_body(bad), timeout=10.0)
                self.assertEqual(
                    status,
                    400,
                    f"expected HTTP 400 for frame_selection={bad!r}, got {status}: {raw!r}",
                )
                err = json.loads(raw)
                self.assertIn("error", err)

            # Non-finite NaN / Infinity in raw JSON also rejected with HTTP 400
            for raw_num in ("NaN", "Infinity", "-Infinity"):
                raw_payload = (
                    '{"model":"qwen3-native","stream":false,"messages":[{"role":"user","content":['
                    '{"type":"video_url","video_url":{"url":"'
                    + video_data_url
                    + '","frame_selection":{"policy":"temporal","motion_threshold":'
                    + raw_num
                    + '}}},{"type":"text","text":"Describe the clip."}]}]}'
                ).encode("utf-8")
                status, _, raw = showcase.http_request(endpoint, raw_payload, timeout=10.0)
                self.assertEqual(
                    status,
                    400,
                    f"expected HTTP 400 for raw {raw_num}, got {status}: {raw!r}",
                )
        finally:
            self._stop_mock_server(server, thread)

    def test_validate_behavior_logprobs_evidence_unit_and_fail_closed(self) -> None:
        valid_payload = {
            "format": "umminfer-behavior-logprobs-v1",
            "tokenizer_sha256": showcase.MOCK_TOKENIZER_SHA256,
            "token_ids": [101, 202, 303, 404, 505],
            "logprobs": [-0.1, -0.25, -0.05, -0.4, -0.01],
            "mask": [1, 0, 1, 0, 1],
        }

        # Valid exposed (case-insensitive header + whitespace)
        for header in ("exposed", "EXPOSED", "  Exposed  "):
            res = showcase.validate_behavior_logprobs_evidence(header, valid_payload)
            self.assertEqual(res["status"], "exposed")
            self.assertEqual(res["format"], "umminfer-behavior-logprobs-v1")
            self.assertEqual(res["tokenizer_sha256"], showcase.MOCK_TOKENIZER_SHA256)
            self.assertEqual(res["token_count"], 5)
            self.assertEqual(res["supervised_count"], 3)
            self.assertEqual(
                showcase.format_behavior_logprobs_summary(res),
                "Behavior: 5 tok (3 sup)",
            )

        # Valid not_exposed (case-insensitive header, None payload)
        for header in ("not_exposed", "Not_Exposed", "  NOT_EXPOSED "):
            res = showcase.validate_behavior_logprobs_evidence(header, None)
            self.assertEqual(res["status"], "not_exposed")
            self.assertIsNone(res["format"])
            self.assertIsNone(res["tokenizer_sha256"])
            self.assertIsNone(res["token_count"])
            self.assertIsNone(res["supervised_count"])
            self.assertEqual(
                showcase.format_behavior_logprobs_summary(res),
                "Behavior: not_exposed",
            )

        # Valid absent (None, empty string, whitespace, literal "absent")
        for header in (None, "", "   ", "absent", "ABSENT"):
            res = showcase.validate_behavior_logprobs_evidence(header, None)
            self.assertEqual(res["status"], "absent")
            self.assertIsNone(res["format"])
            self.assertIsNone(res["tokenizer_sha256"])
            self.assertIsNone(res["token_count"])
            self.assertIsNone(res["supervised_count"])
            self.assertEqual(
                showcase.format_behavior_logprobs_summary(res),
                "Behavior: absent",
            )

        # Invalid header values fail closed
        for bad_header in ("partial", "true", "unknown", 123, True):
            with self.assertRaises(AssertionError):
                showcase.validate_behavior_logprobs_evidence(bad_header, None)

        # exposed with missing / malformed payloads fails closed
        bad_exposed_payloads = [
            None,
            "exposed",
            [],
            {**valid_payload, "format": "other-v1"},
            {**valid_payload, "format": None},
            {**valid_payload, "tokenizer_sha256": ""},
            {**valid_payload, "tokenizer_sha256": "   "},
            {**valid_payload, "tokenizer_sha256": 12345},
            {**valid_payload, "token_ids": []},
            {**valid_payload, "token_ids": [1, 2.5, 3, 4, 5]},
            {**valid_payload, "token_ids": [1, True, 3, 4, 5]},
            {**valid_payload, "token_ids": [1, -1, 3, 4, 5]},
            {**valid_payload, "logprobs": [-0.1, -0.2]},
            {**valid_payload, "logprobs": [-0.1, math.nan, -0.05, -0.4, -0.01]},
            {**valid_payload, "logprobs": [-0.1, math.inf, -0.05, -0.4, -0.01]},
            {**valid_payload, "logprobs": [-0.1, -math.inf, -0.05, -0.4, -0.01]},
            {**valid_payload, "logprobs": [-0.1, True, -0.05, -0.4, -0.01]},
            {**valid_payload, "logprobs": [-0.1, "-0.2", -0.05, -0.4, -0.01]},
            {**valid_payload, "mask": [1, 0, 1]},
            {**valid_payload, "mask": [1, 2, 1, 0, 1]},
            {**valid_payload, "mask": [1, -1, 1, 0, 1]},
            {**valid_payload, "mask": [1, True, 1, 0, 1]},
            {**valid_payload, "mask": [1, False, 1, 0, 1]},
        ]
        for bad_payload in bad_exposed_payloads:
            with self.assertRaises(AssertionError, msg=f"accepted bad exposed payload: {bad_payload!r}"):
                showcase.validate_behavior_logprobs_evidence("exposed", bad_payload)

        # not_exposed and absent must reject stray behavior payloads (zero fabrication/mismatch)
        for header in ("not_exposed", None, "", "absent"):
            with self.assertRaises(AssertionError, msg=f"accepted stray payload for header={header!r}"):
                showcase.validate_behavior_logprobs_evidence(header, valid_payload)

        # Non-final SSE frame carrying behavior payload fails closed
        bad_stream = self._sample_stream()
        bad_stream["observations"][0]["frame"]["behavior"] = copy.deepcopy(valid_payload)
        bad_stream["behavior_evidence_header"] = "exposed"
        with self.assertRaisesRegex(AssertionError, "non-final SSE frame"):
            showcase.validate_narrative_stream(
                bad_stream,
                require_audio=True,
                require_mid_stream_slot=True,
                require_behavior_logprobs=True,
            )

    def test_behavior_logprobs_opt_in_cli_and_default_off_wire_contract(self) -> None:
        # 1. Opt-in CLI --self-check --behavior-logprobs exercises exposed + not_exposed + Journey 2 N/A
        with tempfile.TemporaryDirectory(prefix="showcase-beh-cli-") as tmp:
            out_dir = Path(tmp)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(SCRIPT_PATH),
                    "--self-check",
                    "--behavior-logprobs",
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"stderr={proc.stderr}\nstdout={proc.stdout}")
            self.assertIn("SELF-CHECK PASS", proc.stdout)
            self.assertIn("5/5 journeys passed", proc.stdout)

            report = json.loads((out_dir / "showcase_report.json").read_text(encoding="utf-8"))
            md_text = (out_dir / "showcase_report.md").read_text(encoding="utf-8")
            verified = showcase.verify_sha256sums_file(out_dir, out_dir / "SHA256SUMS")
            self.assertGreaterEqual(verified, 11)

            top_beh = report["behavior_logprobs"]
            self.assertTrue(top_beh["enabled"])
            self.assertEqual(top_beh["status"], "partial")
            self.assertEqual(top_beh["requests"], 7)
            self.assertEqual(top_beh["exposed_requests"], 1)
            self.assertEqual(top_beh["not_exposed_requests"], 6)
            self.assertEqual(top_beh["absent_requests"], 0)
            self.assertEqual(top_beh["token_count"], 8)
            self.assertEqual(top_beh["supervised_count"], 8)

            # Journey 1, 3, 4, 5 carry validated behavior_logprobs; Journey 2 does not.
            self.assertEqual(report["journeys"]["journey_1"]["behavior_logprobs"]["status"], "not_exposed")
            self.assertIsNone(report["journeys"]["journey_1"]["behavior_logprobs"]["token_count"])
            self.assertNotIn("behavior_logprobs", report["journeys"]["journey_2"])

            j3 = report["journeys"]["journey_3"]
            self.assertEqual(j3["behavior_logprobs"]["status"], "not_exposed")
            self.assertEqual(j3["behavior_logprobs"]["requests"], 3)
            self.assertIsNone(j3["behavior_logprobs"]["token_count"])
            for turn in j3["turns"]:
                self.assertEqual(turn["behavior_logprobs"]["status"], "not_exposed")
                self.assertIsNone(turn["behavior_logprobs"]["token_count"])

            j4 = report["journeys"]["journey_4"]
            self.assertEqual(j4["pure_text"]["behavior_logprobs"]["status"], "exposed")
            self.assertEqual(j4["pure_text"]["behavior_logprobs"]["token_count"], 8)
            self.assertEqual(j4["pure_text"]["behavior_logprobs"]["supervised_count"], 8)
            self.assertEqual(j4["visual_scene"]["behavior_logprobs"]["status"], "not_exposed")
            self.assertIsNone(j4["visual_scene"]["behavior_logprobs"]["token_count"])
            self.assertEqual(j4["behavior_logprobs"]["status"], "partial")

            self.assertEqual(report["journeys"]["journey_5"]["behavior_logprobs"]["status"], "not_exposed")

            self.assertIn("## Behavior Logprobs Evidence", md_text)
            self.assertIn("Behavior: 8 tok (8 sup)", md_text)
            self.assertIn("Behavior: not_exposed", md_text)
            self.assertIn("N/A (non-chat endpoint)", md_text)
            self.assertNotIn("0 tok", md_text)

        # 2. Combined --native-allocator-telemetry + --behavior-logprobs works across all 5 journeys
        with tempfile.TemporaryDirectory(prefix="showcase-beh-tel-") as tmp:
            report = showcase.run_self_check(
                MANIFEST_PATH,
                out_dir=Path(tmp),
                native_allocator_telemetry=True,
                behavior_logprobs=True,
            )
            self.assertTrue(report["summary"]["all_passed"])
            self.assertEqual(report["native_allocator_telemetry"]["coverage"], "COVERED")
            self.assertEqual(report["behavior_logprobs"]["status"], "partial")
            self.assertEqual(report["journeys"]["journey_3"]["native_allocator_telemetry"]["requests"], 3)
            self.assertEqual(report["journeys"]["journey_3"]["behavior_logprobs"]["requests"], 3)

    def test_behavior_logprobs_exposed_absent_and_fail_closed_mock_server(self) -> None:
        # 1. All-exposed with mixed 0/1 mask
        mixed_payload = showcase._mock_behavior_payload(
            token_ids=[11, 22, 33, 44, 55],
            logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
            mask=[1, 0, 1, 1, 0],
        )
        with tempfile.TemporaryDirectory(prefix="showcase-beh-exposed-") as tmp:
            out_dir = Path(tmp)
            report = showcase.run_self_check(
                MANIFEST_PATH,
                out_dir=out_dir,
                journeys=[1, 3, 4, 5],
                behavior_logprobs=True,
                behavior_factory=lambda _req, _default: ("EXPOSED", copy.deepcopy(mixed_payload)),
            )
            top_beh = report["behavior_logprobs"]
            self.assertEqual(top_beh["status"], "exposed")
            self.assertEqual(top_beh["requests"], 7)
            self.assertEqual(top_beh["exposed_requests"], 7)
            self.assertEqual(top_beh["token_count"], 35)
            self.assertEqual(top_beh["supervised_count"], 21)
            md_text = (out_dir / "showcase_report.md").read_text(encoding="utf-8")
            self.assertIn("Behavior: 5 tok (3 sup)", md_text)
            self.assertIn("Behavior: 35 tok (21 sup)", md_text)

        # 2. All-absent header reports honest Behavior: absent with null counts
        with tempfile.TemporaryDirectory(prefix="showcase-beh-absent-") as tmp:
            out_dir = Path(tmp)
            report = showcase.run_self_check(
                MANIFEST_PATH,
                out_dir=out_dir,
                journeys=[1, 3],
                behavior_logprobs=True,
                behavior_factory=lambda _req, _default: (None, None),
            )
            top_beh = report["behavior_logprobs"]
            self.assertEqual(top_beh["status"], "absent")
            self.assertEqual(top_beh["absent_requests"], 4)
            self.assertIsNone(top_beh["token_count"])
            self.assertIsNone(top_beh["supervised_count"])
            md_text = (out_dir / "showcase_report.md").read_text(encoding="utf-8")
            self.assertIn("Behavior: absent", md_text)
            self.assertNotIn("0 tok", md_text)

        # 3. End-to-end fail-closed rejections on mismatched/malformed server responses
        fail_cases = [
            ("exposed", None, "omitted or returned non-object"),
            ("exposed", {**mixed_payload, "format": "wrong-v1"}, "behavior.format"),
            ("exposed", {**mixed_payload, "mask": [1, 2, 1, 1, 0]}, "behavior.mask"),
            ("not_exposed", copy.deepcopy(mixed_payload), "not_exposed"),
            (None, copy.deepcopy(mixed_payload), "absent"),
        ]
        for hdr, payload, expected_re in fail_cases:
            with self.assertRaisesRegex(AssertionError, expected_re):
                showcase.run_self_check(
                    MANIFEST_PATH,
                    journeys=[1],
                    behavior_logprobs=True,
                    behavior_factory=lambda _req, _default, h=hdr, p=payload: (h, p),
                )

        # 4. Mock server rejects invalid request-side behavior objects & non-stream behavior.logprobs with HTTP 400
        video_data_url = (
            "data:video/mp4;base64,"
            + base64.b64encode(showcase.MOCK_MP4_J2).decode("ascii")
        )
        server, thread, base_url = self._start_mock_server()
        try:
            endpoint = base_url + "/v1/chat/completions"
            bad_bodies = [
                {
                    "model": "qwen3-native",
                    "stream": False,
                    "behavior": {"logprobs": True},
                    "messages": [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": video_data_url}}, {"type": "text", "text": "hi"}]}],
                },
                {
                    "model": "qwen3-native",
                    "stream": True,
                    "behavior": "logprobs",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                {
                    "model": "qwen3-native",
                    "stream": True,
                    "behavior": {"logprobs": True, "extra": 1},
                    "messages": [{"role": "user", "content": "hi"}],
                },
                {
                    "model": "qwen3-native",
                    "stream": True,
                    "behavior": {"logprobs": "yes"},
                    "messages": [{"role": "user", "content": "hi"}],
                },
            ]
            for bad_body in bad_bodies:
                status, _, raw = showcase.http_request(endpoint, bad_body, timeout=10.0)
                self.assertEqual(status, 400, f"expected HTTP 400 for {bad_body!r}, got {status}: {raw!r}")
        finally:
            self._stop_mock_server(server, thread)

    def _assert_macos_session_export_invariants(self, doc: dict) -> None:
        """Mirror clients/ummmlx/Sources/ummmlx/SessionExport.swift importJSON strict checks."""
        self.assertEqual(doc["schema"], "umminfer-session-export-v1")
        self.assertEqual(doc["client"], "umminfer-showcase")
        self.assertIsInstance(doc["exported_at"], str)
        self.assertTrue(doc["exported_at"].strip())
        self.assertIsInstance(doc["session_id"], str)
        self.assertTrue(doc["session_id"].strip())
        self.assertNotIn("messages", doc)
        self.assertIsInstance(doc["turns"], list)
        self.assertEqual(doc["turn_count"], len(doc["turns"]))
        sha_re = re.compile(r"^[0-9a-f]{64}$")

        for idx, turn in enumerate(doc["turns"]):
            self.assertEqual(turn["index"], idx + 1)
            self.assertIn(turn.get("role"), (None, "turn", "user", "assistant"))
            self.assertIsInstance(turn["prompt"], str)
            self.assertIsInstance(turn["response"], str)
            self.assertTrue(turn["asr_text"] is None or isinstance(turn["asr_text"], str))
            self.assertIsInstance(turn["interrupted"], bool)
            self.assertTrue(
                turn["spoken_bytes"] is None
                or (showcase.is_int(turn["spoken_bytes"]) and turn["spoken_bytes"] >= 0)
            )
            self.assertIsInstance(turn["visual_slots"], list)
            for slot in turn["visual_slots"]:
                self.assertIsInstance(slot["id"], str)
                self.assertTrue(slot["id"].strip())
                self.assertEqual(slot["slot_id"], slot["id"])
                self.assertTrue(showcase.is_int(slot["slot_index"]) and slot["slot_index"] >= 0)
                self.assertIn(slot["kind"], ("image", "video"))
                if slot.get("bytes") is not None:
                    self.assertTrue(showcase.is_int(slot["bytes"]) and slot["bytes"] >= 0)
                if slot.get("sha256") is not None:
                    self.assertRegex(slot["sha256"], sha_re)
                if "continuity_mode" in slot:
                    self.assertIn(slot["continuity_mode"], ("prompt_anchor", "i2v_chain"))
                if "anchor_slot_index" in slot:
                    self.assertTrue(
                        showcase.is_int(slot["anchor_slot_index"]) and slot["anchor_slot_index"] >= 0
                    )
                if "anchor_explicit" in slot:
                    self.assertIsInstance(slot["anchor_explicit"], bool)
                if "anchor_sha256" in slot:
                    self.assertRegex(slot["anchor_sha256"], sha_re)

            tel = turn["telemetry"]
            self.assertIsInstance(tel, dict)
            for ms_key in ("ttft_ms", "ttfa_ms", "first_visual_ms", "total_ms"):
                val = tel[ms_key]
                if val is not None:
                    self.assertFalse(isinstance(val, bool))
                    self.assertIsInstance(val, (int, float))
                    self.assertTrue(math.isfinite(val) and val >= 0.0)
            self.assertTrue(showcase.is_int(tel["visual_slots"]) and tel["visual_slots"] >= 0)
            self.assertTrue(showcase.is_int(tel["i2v_count"]) and tel["i2v_count"] >= 0)

            na = tel["native_allocator"]
            if na is not None:
                self.assertIn(na["coverage"], ("COVERED", "PARTIAL", "NOT_COVERED"))
                self.assertIsInstance(na["covered"], bool)
                self.assertTrue(showcase.is_int(na["covered_stage_count"]))
                self.assertTrue(showcase.is_int(na["total_stage_count"]))
                self.assertLessEqual(na["covered_stage_count"], na["total_stage_count"])
                if na["covered"]:
                    self.assertNotEqual(na["coverage"], "NOT_COVERED")
                    self.assertTrue(showcase.is_int(na["peak_bytes"]) and na["peak_bytes"] >= 0)
                else:
                    self.assertEqual(na["coverage"], "NOT_COVERED")
                    self.assertIsNone(na["peak_bytes"])
                self.assertIsInstance(na["label"], str)
                self.assertIsInstance(na["details"], list)
                for line in na["details"]:
                    self.assertIsInstance(line, str)

            beh = tel["behavior_logprobs"]
            if beh is not None:
                self.assertIn(beh["status"], ("exposed", "not_exposed", "absent"))
                self.assertIsInstance(beh["label"], str)
                if beh["status"] == "exposed":
                    self.assertTrue(showcase.is_int(beh["token_count"]) and beh["token_count"] >= 0)
                    self.assertTrue(
                        showcase.is_int(beh["supervised_count"])
                        and 0 <= beh["supervised_count"] <= beh["token_count"]
                    )
                else:
                    self.assertIsNone(beh["token_count"])
                    self.assertIsNone(beh["supervised_count"])

    def _validate_sessions_with_web_ui_node(self, session_paths: list[Path]) -> list[dict]:
        """Validate exported JSON files directly against web/index.html parseImportedSessionJson."""
        node = shutil.which("node")
        self.assertIsNotNone(node, "node is required to validate GUI sessions against web/index.html")
        web_html = (REPO_ROOT / "web" / "index.html").read_text(encoding="utf-8")
        blocks = []
        for begin_marker, end_marker in (
            ("// TURN_TELEMETRY_BEGIN", "// TURN_TELEMETRY_END"),
            ("// NARRATIVE_CONTINUITY_BEGIN", "// NARRATIVE_CONTINUITY_END"),
            ("// REALTIME_AUDIO_METER_BEGIN", "// REALTIME_AUDIO_METER_END"),
            ("// SESSION_EXPORT_BEGIN", "// SESSION_EXPORT_END"),
            ("// SESSION_IMPORT_BEGIN", "// SESSION_IMPORT_END"),
        ):
            self.assertIn(begin_marker, web_html)
            self.assertIn(end_marker, web_html)
            blocks.append(web_html.split(begin_marker, 1)[1].split(end_marker, 1)[0])

        harness = "\n".join(blocks) + r"""
const fs = require('fs');
globalThis.fetch = () => { throw new Error('unexpected fetch during session import'); };
globalThis.WebSocket = class ForbiddenWebSocket {
  constructor() { throw new Error('unexpected WebSocket during session import'); }
};
const paths = JSON.parse(process.argv[1]);
const summaries = [];
for (const filePath of paths) {
  const raw = fs.readFileSync(filePath, 'utf8');
  const parsed = parseImportedSessionJson(raw);
  const state = { turns: [], history: [], sessionID: '', status: '', chip: null };
  const ctx = {
    isBusy: () => false,
    stopActiveAudio() {},
    resetState(nextId) { state.sessionID = nextId; state.turns = []; state.history = []; },
    clearTranscript() {},
    clearError() {},
    addTurn(role, text) {
      return { role, text, asrText: '', asrEl: { textContent: '' }, slots: [], telemetry: null, pills: [] };
    },
    renderAssistantText(a, text) { a.renderedText = text; },
    addImportedVisualSlot(a, slotMeta) { a.slots.push(slotMeta); },
    attachInterruptedBadge() {},
    renderTelemetry(a) { a.pills = formatTurnTelemetry(a.telemetry).map((p) => p.label); },
    recordTurn(u, a, p, r) { state.turns.push({ u, a }); if (p) state.history.push(p); if (r) state.history.push(r); },
    setSessionChip(chip) { state.chip = chip; },
    syncUI() {},
    setStatus(msg) { state.status = msg; },
  };
  const applied = applyImportedSessionWithContext(parsed, ctx);
  summaries.push({
    file: filePath,
    schema: applied.schema,
    session_id: applied.sessionId,
    turn_count: applied.turnCount,
    slot_counts: state.turns.map((t) => (t.a ? t.a.slots.length : 0)),
    pills: state.turns.map((t) => (t.a ? t.a.pills : [])),
  });
}
process.stdout.write(JSON.stringify(summaries));
"""
        proc = subprocess.run(
            [node, "-e", harness, json.dumps([str(p) for p in session_paths])],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"web/index.html parseImportedSessionJson rejected exported session(s):\n"
            f"stderr={proc.stderr}\nstdout={proc.stdout}",
        )
        return json.loads(proc.stdout)

    def test_gui_session_export_default_off_unchanged(self) -> None:
        with tempfile.TemporaryDirectory(prefix="showcase-gui-off-") as tmp:
            out_dir = Path(tmp)
            proc = subprocess.run(
                [sys.executable, "-I", str(SCRIPT_PATH), "--self-check", "--out-dir", str(out_dir)],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"stderr={proc.stderr}\nstdout={proc.stdout}")
            self.assertFalse((out_dir / "gui_sessions").exists())
            report = json.loads((out_dir / "showcase_report.json").read_text(encoding="utf-8"))
            md_text = (out_dir / "showcase_report.md").read_text(encoding="utf-8")
            sums_text = (out_dir / "SHA256SUMS").read_text(encoding="utf-8")
            self.assertNotIn("gui_sessions", report)
            self.assertNotIn("## GUI Session Exports", md_text)
            self.assertNotIn("gui_sessions/", sums_text)

    def test_gui_session_export_cli_self_check_and_cross_client_validation(self) -> None:
        forbidden = [
            "h" + "20b",
            "/" + "Users/",
            "/" + "home/",
            "byte" + "dance",
            ";base64,",
            "data:image/",
            "data:video/",
            "data:audio/",
        ]

        # 1. Baseline --self-check --export-gui-sessions (all 5 journeys)
        with tempfile.TemporaryDirectory(prefix="showcase-gui-base-") as tmp:
            out_dir = Path(tmp)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(SCRIPT_PATH),
                    "--self-check",
                    "--export-gui-sessions",
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"stderr={proc.stderr}\nstdout={proc.stdout}")
            self.assertIn("SELF-CHECK PASS", proc.stdout)

            report = json.loads((out_dir / "showcase_report.json").read_text(encoding="utf-8"))
            md_text = (out_dir / "showcase_report.md").read_text(encoding="utf-8")
            sums_path = out_dir / "SHA256SUMS"
            verified = showcase.verify_sha256sums_file(out_dir, sums_path)
            self.assertGreaterEqual(verified, 16)

            gui_meta = report["gui_sessions"]
            self.assertEqual(
                [item["journey_id"] for item in gui_meta],
                ["journey_1", "journey_2", "journey_3", "journey_4", "journey_5"],
            )
            self.assertEqual([item["turn_count"] for item in gui_meta], [1, 1, 3, 2, 1])
            self.assertIn("## GUI Session Exports", md_text)

            sums_text = sums_path.read_text(encoding="utf-8")
            session_paths: list[Path] = []
            docs_by_id: dict[str, dict] = {}
            for item in gui_meta:
                rel_path = item["path"]
                self.assertEqual(rel_path, f"gui_sessions/{item['journey_id']}.session.json")
                self.assertIn(f"  {rel_path}", sums_text)
                self.assertIn(f"`{rel_path}`", md_text)
                file_path = out_dir / rel_path
                self.assertTrue(file_path.is_file())
                raw_bytes = file_path.read_bytes()
                self.assertEqual(len(raw_bytes), item["bytes"])
                self.assertEqual(showcase.sha256_bytes(raw_bytes), item["sha256"])
                raw_text = raw_bytes.decode("utf-8")
                self.assertTrue(raw_text.endswith("\n"))
                lowered = raw_text.lower()
                for token in forbidden:
                    self.assertNotIn(token.lower(), lowered, f"forbidden token {token!r} in {rel_path}")
                doc = json.loads(raw_text)
                self._assert_macos_session_export_invariants(doc)
                session_paths.append(file_path)
                docs_by_id[item["journey_id"]] = doc

            # Journey 1: interleaved narrative + audio + 2 visual slots, null opt-in telemetry
            j1_turn = docs_by_id["journey_1"]["turns"][0]
            self.assertTrue(j1_turn["prompt"])
            self.assertTrue(j1_turn["response"])
            self.assertEqual(len(j1_turn["visual_slots"]), 2)
            self.assertIsNotNone(j1_turn["telemetry"]["ttft_ms"])
            self.assertIsNotNone(j1_turn["telemetry"]["ttfa_ms"])
            self.assertIsNone(j1_turn["telemetry"]["native_allocator"])
            self.assertIsNone(j1_turn["telemetry"]["behavior_logprobs"])

            # Journey 2: I2V render with honest null ttft/ttfa/native_allocator/behavior_logprobs
            j2_turn = docs_by_id["journey_2"]["turns"][0]
            self.assertEqual(j2_turn["response"], "")
            self.assertEqual(len(j2_turn["visual_slots"]), 1)
            self.assertEqual(j2_turn["visual_slots"][0]["kind"], "video")
            self.assertEqual(j2_turn["visual_slots"][0]["continuity_mode"], "i2v_chain")
            self.assertTrue(j2_turn["visual_slots"][0]["anchor_explicit"])
            self.assertEqual(
                j2_turn["visual_slots"][0]["anchor_sha256"],
                report["journeys"]["journey_2"]["source_image_sha256"],
            )
            self.assertIsNone(j2_turn["telemetry"]["ttft_ms"])
            self.assertIsNone(j2_turn["telemetry"]["ttfa_ms"])
            self.assertEqual(j2_turn["telemetry"]["i2v_count"], 1)
            self.assertIsNone(j2_turn["telemetry"]["native_allocator"])
            self.assertIsNone(j2_turn["telemetry"]["behavior_logprobs"])

            # Journey 3: 3-turn video QA -> 3 turns, zero base64, honest null ttft/ttfa
            j3_doc = docs_by_id["journey_3"]
            self.assertEqual(j3_doc["turn_count"], 3)
            for t in j3_doc["turns"]:
                self.assertTrue(t["prompt"])
                self.assertTrue(t["response"])
                self.assertEqual(t["visual_slots"], [])
                self.assertIsNone(t["telemetry"]["ttft_ms"])
                self.assertIsNone(t["telemetry"]["ttfa_ms"])
                self.assertIsNotNone(t["telemetry"]["total_ms"])

            # Journey 4: abstention vs visual scene -> 2 turns
            j4_turns = docs_by_id["journey_4"]["turns"]
            self.assertEqual(len(j4_turns), 2)
            self.assertEqual(j4_turns[0]["visual_slots"], [])
            self.assertTrue(j4_turns[0]["response"])
            self.assertEqual(j4_turns[0]["telemetry"]["visual_slots"], 0)
            self.assertIsNone(j4_turns[0]["telemetry"]["first_visual_ms"])
            self.assertEqual(len(j4_turns[1]["visual_slots"]), 2)
            self.assertTrue(j4_turns[1]["response"])
            self.assertEqual(j4_turns[1]["telemetry"]["visual_slots"], 2)

            # Journey 5: storyboard I2V chain
            j5_turn = docs_by_id["journey_5"]["turns"][0]
            self.assertEqual(len(j5_turn["visual_slots"]), 2)
            self.assertEqual(j5_turn["visual_slots"][1]["continuity_mode"], "i2v_chain")
            self.assertEqual(j5_turn["visual_slots"][1]["anchor_slot_index"], 0)
            self.assertGreaterEqual(j5_turn["telemetry"]["i2v_count"], 1)

            # Determinism: exporting again yields identical SHA-256 and bytes
            with tempfile.TemporaryDirectory(prefix="showcase-gui-det-") as tmp2:
                out2 = Path(tmp2)
                r1 = showcase.export_gui_sessions(report, out2)
                r2 = showcase.export_gui_sessions(report, out2)
                self.assertEqual(r1, r2)

            # Cross-client Node validation directly against web/index.html parseImportedSessionJson
            web_results = self._validate_sessions_with_web_ui_node(session_paths)
            self.assertEqual(len(web_results), 5)
            self.assertEqual([r["turn_count"] for r in web_results], [1, 1, 3, 2, 1])
            self.assertEqual([r["slot_counts"] for r in web_results], [[2], [1], [0, 0, 0], [0, 2], [2]])

        # 2. Combined --export-gui-sessions + --native-allocator-telemetry + --behavior-logprobs
        with tempfile.TemporaryDirectory(prefix="showcase-gui-full-") as tmp:
            out_dir = Path(tmp)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(SCRIPT_PATH),
                    "--self-check",
                    "--export-gui-sessions",
                    "--native-allocator-telemetry",
                    "--behavior-logprobs",
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"stderr={proc.stderr}\nstdout={proc.stdout}")
            report = json.loads((out_dir / "showcase_report.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(showcase.verify_sha256sums_file(out_dir, out_dir / "SHA256SUMS"), 16)

            full_paths = [out_dir / item["path"] for item in report["gui_sessions"]]
            full_docs = {
                item["journey_id"]: json.loads((out_dir / item["path"]).read_text(encoding="utf-8"))
                for item in report["gui_sessions"]
            }
            for doc in full_docs.values():
                self._assert_macos_session_export_invariants(doc)

            # Journey 1 has COVERED VRAM + not_exposed behavior (null counts)
            j1_tel = full_docs["journey_1"]["turns"][0]["telemetry"]
            self.assertEqual(j1_tel["native_allocator"]["coverage"], "COVERED")
            self.assertTrue(j1_tel["native_allocator"]["covered"])
            self.assertEqual(j1_tel["native_allocator"]["peak_bytes"], 301989888)
            self.assertEqual(j1_tel["behavior_logprobs"]["status"], "not_exposed")
            self.assertIsNone(j1_tel["behavior_logprobs"]["token_count"])
            self.assertIsNone(j1_tel["behavior_logprobs"]["supervised_count"])

            # Journey 2 stays honestly null for both
            j2_tel = full_docs["journey_2"]["turns"][0]["telemetry"]
            self.assertIsNone(j2_tel["native_allocator"])
            self.assertIsNone(j2_tel["behavior_logprobs"])

            # Journey 3 has per-turn COVERED VRAM + not_exposed behavior on all 3 turns
            for turn in full_docs["journey_3"]["turns"]:
                self.assertEqual(turn["telemetry"]["native_allocator"]["coverage"], "COVERED")
                self.assertEqual(turn["telemetry"]["behavior_logprobs"]["status"], "not_exposed")
                self.assertIsNone(turn["telemetry"]["behavior_logprobs"]["token_count"])

            # Journey 4 contrasts Turn 1 (exposed, 8 tok / 8 sup) vs Turn 2 (not_exposed, null counts)
            j4_t1_beh = full_docs["journey_4"]["turns"][0]["telemetry"]["behavior_logprobs"]
            j4_t2_beh = full_docs["journey_4"]["turns"][1]["telemetry"]["behavior_logprobs"]
            self.assertEqual(j4_t1_beh["status"], "exposed")
            self.assertEqual(j4_t1_beh["token_count"], 8)
            self.assertEqual(j4_t1_beh["supervised_count"], 8)
            self.assertEqual(j4_t1_beh["format"], "umminfer-behavior-logprobs-v1")
            self.assertEqual(j4_t2_beh["status"], "not_exposed")
            self.assertIsNone(j4_t2_beh["token_count"])
            self.assertIsNone(j4_t2_beh["supervised_count"])

            web_full = self._validate_sessions_with_web_ui_node(full_paths)
            self.assertEqual(len(web_full), 5)
            self.assertTrue(any("Behavior: 8 tok (8 sup)" in p for p in web_full[3]["pills"][0]))
            self.assertTrue(any("Behavior: not_exposed" in p for p in web_full[3]["pills"][1]))
            self.assertTrue(any("VRAM: 288 MiB peak" in p for p in web_full[0]["pills"][0]))

        # 3. Honest NOT_COVERED + absent telemetry (zero fabricated zeros)
        def _not_covered_factory(_req, request_id, stage_specs):
            stages = [
                showcase._mock_telemetry_stage(
                    stage_name,
                    request_id=request_id,
                    coverage="NOT_COVERED",
                    reason="batched_request_attribution_unavailable",
                )
                for stage_name, _ in stage_specs
            ]
            return showcase._mock_telemetry_envelope(request_id, stages)

        with tempfile.TemporaryDirectory(prefix="showcase-gui-nc-") as tmp:
            out_dir = Path(tmp)
            report = showcase.run_self_check(
                MANIFEST_PATH,
                out_dir=out_dir,
                native_allocator_telemetry=True,
                telemetry_factory=_not_covered_factory,
                behavior_logprobs=True,
                behavior_factory=lambda _req, _default: (None, None),
                export_gui_sessions=True,
            )
            nc_paths = [out_dir / item["path"] for item in report["gui_sessions"]]
            for item in report["gui_sessions"]:
                doc = json.loads((out_dir / item["path"]).read_text(encoding="utf-8"))
                self._assert_macos_session_export_invariants(doc)
                if item["journey_id"] != "journey_2":
                    for turn in doc["turns"]:
                        na = turn["telemetry"]["native_allocator"]
                        beh = turn["telemetry"]["behavior_logprobs"]
                        self.assertEqual(na["coverage"], "NOT_COVERED")
                        self.assertFalse(na["covered"])
                        self.assertIsNone(na["peak_bytes"])
                        self.assertEqual(na["label"], "VRAM: N/C")
                        self.assertEqual(beh["status"], "absent")
                        self.assertIsNone(beh["token_count"])
                        self.assertIsNone(beh["supervised_count"])
            web_nc = self._validate_sessions_with_web_ui_node(nc_paths)
            self.assertEqual(len(web_nc), 5)
            self.assertTrue(any("VRAM: N/C" in p for p in web_nc[0]["pills"][0]))
            self.assertTrue(any("Behavior: absent" in p for p in web_nc[0]["pills"][0]))

    def test_build_gui_session_export_unit_scrubbing_anchors_and_fail_closed(self) -> None:
        source_sha = showcase.sha256_bytes(showcase.MOCK_PNG_J5)
        video_sha = showcase.sha256_bytes(showcase.MOCK_MP4_J5)
        inline_uri = "data:image/png;base64," + base64.b64encode(showcase.MOCK_PNG_J5).decode("ascii")
        j5_result = {
            "journey_id": "storyboard_i2v_chain",
            "passed": True,
            "prose": f"Clean scene with scrubbed {inline_uri} payload.",
            "ttft_s": 0.015,
            "ttfa_s": 0.045,
            "first_visual_slot_s": 0.02,
            "total_wall_s": 0.25,
            "slots": [
                {
                    "slot_id": "slot-0",
                    "slot_index": 0,
                    "media_type": "image",
                    "prompt": f"Keyframe {inline_uri}",
                    "bytes": len(showcase.MOCK_PNG_J5),
                    "sha256": source_sha,
                    "continuity_mode": "i2v_chain",
                },
                {
                    "slot_id": "slot-1",
                    "slot_index": 1,
                    "media_type": "video",
                    "prompt": "Smooth camera pan",
                    "bytes": len(showcase.MOCK_MP4_J5),
                    "sha256": video_sha,
                    "continuity_mode": "i2v_chain",
                    "anchor_slot_index": 0,
                    "anchor_explicit": True,
                    "anchor_sha256": source_sha,
                },
            ],
        }
        doc = showcase.build_gui_session_export(
            j5_result,
            model="qwen3-native",
            exported_at="2026-11-26T10:15:00.000Z",
            prompt=f"Prompt with {inline_uri}",
        )
        self._assert_macos_session_export_invariants(doc)
        self.assertEqual(doc["exported_at"], "2026-11-26T10:15:00.000Z")
        self.assertEqual(doc["session_id"], "showcase-storyboard_i2v_chain")
        serialized = json.dumps(doc)
        self.assertNotIn(";base64,", serialized)
        self.assertIn("[omitted-base64]", doc["turns"][0]["prompt"])
        self.assertIn("[omitted-base64]", doc["turns"][0]["response"])
        self.assertIn("[omitted-base64]", doc["turns"][0]["visual_slots"][0]["prompt"])
        self.assertTrue(doc["turns"][0]["visual_slots"][1]["anchor_explicit"])
        self.assertEqual(doc["turns"][0]["visual_slots"][1]["anchor_sha256"], source_sha)

        # Fail closed on malformed inputs
        with self.assertRaises(ValueError):
            showcase.build_gui_session_export({"passed": True}, model="qwen3-native")
        with self.assertRaises(ValueError):
            showcase.build_gui_session_export(j5_result, model="qwen3-native", exported_at="   ")


if __name__ == "__main__":
    unittest.main()
