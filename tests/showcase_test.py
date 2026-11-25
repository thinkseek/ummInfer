#!/usr/bin/env python3
"""Unit and integration tests for the North-Star Native Multimodal Showcase Suite."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
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


if __name__ == "__main__":
    unittest.main()
