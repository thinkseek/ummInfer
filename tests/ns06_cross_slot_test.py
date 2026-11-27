#!/usr/bin/env python3
"""Offline frozen-plan and actual Web client lifecycle tests for Task #76 follow-up."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns06_cross_slot_live as live  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def synthetic_record(arm, prereg):
    states = prereg["expected_outcomes"][arm]["visual_leg_states"]
    slots = []
    artifacts = []
    legs = []
    hashes = {index: hashlib.sha256(f"slot-{index}".encode()).hexdigest()
              for index in range(6)}
    for spec, state in zip(prereg["slots"], states):
        index = spec["index"]
        slot_id = f"slot-{index}"
        source = spec["source"]
        slot = {"slot_id": slot_id, "slot_index": index,
                "media_type": spec["kind"], "prompt": f"shot {spec['shot']}",
                "continuity_mode": "i2v_chain"}
        if source is not None:
            slot.update({"anchor_slot_index": source, "anchor_explicit": True})
        slots.append(slot)
        leg = {"slot_id": slot_id, "slot_index": index, "stage": spec["kind"],
               "state": state}
        if state == "failed":
            leg["error"] = "synthetic expected parent rejection"
        if state == "completed":
            item = {"slot_id": slot_id, "slot_index": index,
                    "media_type": spec["kind"], "bytes": 1000 + index,
                    "sha256": hashes[index]}
            if source is not None:
                item.update({"anchor_slot_index": source, "anchor_explicit": True,
                             "anchor_sha256": hashes[source]})
                leg.update({"anchor_slot_index": source, "anchor_explicit": True,
                            "anchor_sha256": hashes[source]})
            artifacts.append(item)
        legs.append(leg)
    return {"session_id": f"ns06-{arm}-76", "clean_text": "Mira enters the cafe.",
            "events": [{"type": "asr_final", "text": "spoken storyboard request"}],
            "visual_slots": slots, "artifacts": artifacts, "terminal_legs": legs}


class FrozenPlanTests(unittest.TestCase):
    def test_frozen_inputs_references_and_claims(self):
        prereg = live.load_prereg(
            ROOT / "benchmarks/cross-slot-visual-continuity/ns06-preregistration.json")
        self.assertEqual([slot["source"] for slot in prereg["slots"]], [None, 0, 0, 2, 0, 4])
        self.assertEqual(prereg["expected_outcomes"]["before"]["keyframe_count"], 1)
        self.assertEqual(prereg["expected_outcomes"]["after"]["keyframe_count"], 3)
        self.assertIn("NOT_RUN", prereg["claim_boundaries"]["ns06_human_character_style_threshold"])
        body = live.main_body(prereg, "before", b"sketch", b"voice")
        self.assertEqual(body["narrative"]["max_slots"], 6)
        self.assertEqual([part["type"] for part in body["messages"][0]["content"]],
                         ["text", "image_url", "input_audio"])
        self.assertTrue(live.trace_leg_completed([
            {"event": "leg_end", "leg": "asr", "status": "completed",
             "request_id": "http-1"}], "http-1", "asr"))
        self.assertFalse(live.trace_leg_completed([
            {"event": "leg_end", "leg": "asr", "status": "completed",
             "request_id": "http-2"}], "http-1", "asr"))


class PortableOutputPathTests(unittest.TestCase):
    def test_both_live_harnesses_require_caller_selected_output(self):
        for name in ("cross_slot_visual_continuity_live.py", "ns06_cross_slot_live.py"):
            with self.subTest(name=name):
                result = subprocess.run(
                    [sys.executable, "-I", str(ROOT / "tests" / name)],
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("--out-dir", result.stderr)
                self.assertIn("requires", result.stderr)
                self.assertNotIn("/tmp/umminfer-", result.stderr)


@unittest.skipUnless(shutil.which("node"), "node is required for production Web helpers")
class ProductionClientLifecycleTests(unittest.TestCase):
    def test_actual_export_clear_import_reexport_before_and_after(self):
        prereg = live.load_prereg(
            ROOT / "benchmarks/cross-slot-visual-continuity/ns06-preregistration.json")
        for arm in ("before", "after"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as tmp:
                result = live.run_client_lifecycle(
                    prereg, arm, synthetic_record(arm, prereg), ROOT / "web",
                    shutil.which("node"), Path(tmp))
                self.assertTrue(result["clear"])
                self.assertTrue(result["exact_reexport"])
                self.assertEqual(result["network_calls"], 0)
                self.assertEqual(result["reconnect_history_length"], 2)
                self.assertEqual(len(result["restored_slots"]), 6)
                self.assertEqual(result["raw_media_and_pcm_restoration"],
                                 "NOT_SUPPORTED_BY_EXISTING_SESSION_SCHEMA")
                out = Path(tmp) / "client" / "out"
                self.assertEqual((out / "export.json").read_bytes(),
                                 (out / "reexport.json").read_bytes())
                restored = result["restored_slots"]
                for spec, slot in zip(prereg["slots"], restored):
                    self.assertEqual(slot["id"], f"slot-{spec['index']}")
                    self.assertEqual(slot.get("anchor_slot_index"), spec["source"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
