#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic frozen-suite, request-shape, and routing-score tests."""
import base64
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import ns03_routing as ns03  # noqa: E402
import ns03_routing_live as live  # noqa: E402
import ns03_routing_verify as verify  # noqa: E402

SUITE_PATH = ROOT / "benchmarks/ns03-routing/suite.json"
PREREG_PATH = ROOT / "benchmarks/ns03-routing/preregistration.json"
HAS_OBSERVER_DEPS = all(importlib.util.find_spec(name) is not None for name in ("msgpack", "zmq"))


def record(images=0, videos=0, *, terminal="complete", errors=None, voice=True, decoded=True):
    kinds = ["image"] * images + ["video"] * videos
    slots = [{"slot_id": f"slot-{i}", "slot_index": i, "media_type": kind, "stage": kind,
              "prompt_sha256": "a" * 64, "text_offset": 0} for i, kind in enumerate(kinds)]
    artifacts = [{"artifact_id": f"art-{i}", "media_type": kind, "mime_type": "image/png" if kind == "image" else "video/mp4",
                  "sha256": str(i) * 64, "bytes": 100, "slot_id": f"slot-{i}", "slot_index": i,
                  "decoded": decoded} for i, kind in enumerate(kinds)]
    return {"status": "captured", "slots": slots, "artifacts": artifacts,
            "errors": list(errors or []), "terminal_kind": terminal,
            "terminal_count": 1 if terminal else 0, "done_count": 1,
            "raw_response_sha256": "b" * 64, "clean_text_sha256": "c" * 64,
            "request_sha256": "d" * 64, "clean_text": "ordinary answer",
            "audio": {"artifact_count": 1 if voice else 0, "bytes": 20 if voice else 0,
                      "pcm16_valid": voice}, "parse_ambiguous": False}


def write_vlm_capture(case_dir, messages, native_request_id="public:vlm"):
    import msgpack
    case_dir.mkdir(parents=True)
    raw = msgpack.packb({"type": "submit", "request_id": native_request_id,
                         "data": {"request": {"inputs": {"messages": messages}}}},
                        use_bin_type=True)
    row = {"observed_monotonic_ns": 1, "request_id": native_request_id,
           "raw_sha256": ns03.sha256_bytes(raw), "raw_base64": base64.b64encode(raw).decode(),
           "messages": verify.normalize_vlm_messages(messages)}
    path = case_dir / "vlm-prompts.jsonl"
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    metadata = {"path": path.name, "rows": 1, "sha256": ns03.sha256_file(path),
                "native_request_id": native_request_id}
    return row, {"vlm_capture": metadata}


class NS03RoutingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = ns03.load_suite(SUITE_PATH)
        cls.prereg = ns03.load_json(PREREG_PATH)
        cls.cases = {case["id"]: case for case in cls.suite["cases"]}

    def test_frozen_populations_multilingual_and_context(self):
        self.assertEqual(len(self.suite["cases"]), 40)
        for population in ns03.POPULATIONS:
            for kind in ns03.CLASSES:
                self.assertEqual(sum(c["population"] == population and c["class"] == kind
                                     for c in self.suite["cases"]), 5)
        self.assertGreaterEqual(len({c["language"] for c in self.suite["cases"]}), 7)
        self.assertGreaterEqual(sum(c.get("adversarial_visual_nouns") is True
                                    for c in self.suite["cases"]), 5)
        self.assertEqual(self.prereg["suite_sha256"], ns03.sha256_bytes(ns03.canonical(self.suite)))
        self.assertEqual(ns03.sha256_file(SUITE_PATH),
                         "b28004b52b9d44cdd8ee238a53c1c8329b9c8b3db6ab4c40ac3c0f07dafbcb9b")
        self.assertEqual(ns03.sha256_file(PREREG_PATH),
                         "ccea4d085a743f8f1e412b43c9cdd22c1b1c3f64e6bb51d7fc66355fbfc20f8e")
        for case in self.suite["cases"]:
            if case["class"] == "followup":
                self.assertTrue(case["setup"]["prompt"])
                self.assertTrue(case["context_any"])

    def test_default_public_request_has_no_manual_routing_override(self):
        case = self.cases["dev-text-01"]
        body = ns03.default_request(case, session_id="frozen-session")
        self.assertEqual(body, {
            "model": "qwen3-native", "messages": [{"role": "user", "content": case["prompt"]}],
            "max_tokens": 256, "temperature": 0, "stream": True, "session_id": "frozen-session",
            "session_history": True,
            "image": {"size": "512x512", "output_format": "png"},
            "video": {"size": "704x480", "output_format": "mp4", "seconds": 2,
                      "num_frames": 29, "steps": 25, "guidance_scale": 5.0},
            "narrative": {"enabled": True, "visual_mode": "auto", "max_slots": 3,
                          "allow_text_only": True},
            "audio": {"voice": "vivian", "format": "wav", "stream": True},
            "modalities": ["text", "image", "video", "audio"],
        })
        followup = ns03.default_request(self.cases["dev-followup-01"],
                                        session_id="frozen-session", followup=True)
        self.assertTrue(followup["session_history"])
        self.assertNotIn("frame_selection", followup)
        self.assertNotIn("seed", followup)

    def test_legacy_initial_history_false_preserves_task78_shape(self):
        text = ns03.default_request(self.cases["dev-text-01"], session_id="legacy",
                                    initial_history=False)
        setup = ns03.default_request(self.cases["dev-followup-01"], session_id="legacy",
                                     initial_history=False)
        answer = ns03.default_request(self.cases["dev-followup-01"], session_id="legacy",
                                      followup=True, initial_history=False)
        self.assertNotIn("session_history", text)
        self.assertNotIn("session_history", setup)
        self.assertIs(answer["session_history"], True)
        self.assertEqual(text, verify.expected_request(self.cases["dev-text-01"], "legacy", False))
        self.assertEqual(setup, verify.expected_request(self.cases["dev-followup-01"], "legacy", True))
        self.assertEqual(answer, verify.expected_request(self.cases["dev-followup-01"], "legacy", False))

    def test_first_turn_history_after_request_matches_independent_shape(self):
        for case, setup, followup in ((self.cases["dev-text-01"], False, False),
                                      (self.cases["dev-followup-01"], True, False),
                                      (self.cases["dev-followup-01"], False, True)):
            actual = ns03.default_request(case, session_id="after", followup=followup,
                                          initial_history=True)
            expected = verify.expected_request(case, "after", setup, initial_history=True)
            self.assertEqual(actual, expected)
            self.assertIs(actual["session_history"], True)

    def test_followup_only_selection_is_complete_and_ordered(self):
        for population in ns03.POPULATIONS:
            selected = live.selected_cases(self.suite, population, "followup")
            self.assertEqual([case["id"] for case in selected],
                             [f"{'dev' if population == 'development' else 'hold'}-followup-{index:02d}"
                              for index in range(1, 6)])

    def test_four_class_golden_decisions(self):
        expectations = {"dev-text-01": (0, 0), "dev-static-01": (1, 0),
                        "dev-dynamic-01": (0, 1)}
        for case_id, (images, videos) in expectations.items():
            result = ns03.evaluate_record(self.cases[case_id], record(images, videos))
            self.assertTrue(result["routing_correct"], result)
            self.assertTrue(result["end_to_end_correct"], result)
            self.assertFalse(result["redundant_visual_case"])

    def test_redundancy_miss_and_wrong_modality_remain_distinct(self):
        extra = ns03.evaluate_record(self.cases["dev-text-01"], record(images=1))
        self.assertFalse(extra["routing_correct"])
        self.assertTrue(extra["redundant_visual_case"])
        missing = ns03.evaluate_record(self.cases["dev-dynamic-01"], record())
        self.assertFalse(missing["routing_correct"])
        self.assertEqual(missing["missing_visual_slots"], 1)
        self.assertFalse(missing["redundant_visual_case"])
        wrong = ns03.evaluate_record(self.cases["dev-static-01"], record(videos=1))
        self.assertFalse(wrong["routing_correct"])
        self.assertTrue(wrong["wrong_modality"])
        self.assertFalse(wrong["redundant_visual_case"])
        self.assertTrue(wrong["redundant_video_case"])
        self.assertFalse(wrong["redundant_image_case"])
        excess = ns03.evaluate_record(self.cases["dev-static-01"], record(images=2))
        self.assertTrue(excess["redundant_visual_case"])
        self.assertEqual(excess["redundant_visual_slots"], 1)

    def test_failed_media_does_not_erase_route_or_voice_metrics(self):
        failed = ns03.evaluate_record(self.cases["dev-static-01"], record(images=1, decoded=False))
        self.assertTrue(failed["routing_correct"])
        self.assertFalse(failed["end_to_end_correct"])
        no_voice = ns03.evaluate_record(self.cases["dev-text-01"], record(voice=False))
        self.assertTrue(no_voice["routing_correct"])
        self.assertFalse(no_voice["end_to_end_correct"])
        self.assertFalse(no_voice["voice_present"])

    def test_followup_requires_real_history_and_context(self):
        case = self.cases["dev-followup-01"]
        answer = record()
        answer["clean_text"] = "Take the product of the known primes and add plus one to it."
        answer["session_history"] = True
        answer["same_session_as_setup"] = True
        setup = record()
        result = ns03.evaluate_record(case, answer, setup)
        self.assertTrue(result["followup_correct"], result["followup"])
        for mutation in ("history", "session", "setup", "context"):
            changed = copy.deepcopy(answer)
            changed_setup = copy.deepcopy(setup)
            if mutation == "history":
                changed["session_history"] = False
            elif mutation == "session":
                changed["same_session_as_setup"] = False
            elif mutation == "setup":
                changed_setup["terminal_kind"] = "error"
            else:
                changed["clean_text"] = "An unrelated response about pictures."
            rejected = ns03.evaluate_record(case, changed, changed_setup)
            self.assertFalse(rejected["followup_correct"], mutation)

    def test_public_leg_errors_are_retained_without_changing_frozen_route_metric(self):
        response = record()
        response["leg_errors"] = [{"stage": "tts", "slot_id": None, "slot_index": None,
                                   "error": "streamed audio exceeds the artifact limit"}]
        result = ns03.evaluate_record(self.cases["dev-text-01"], response)
        self.assertTrue(result["routing_correct"])
        self.assertTrue(result["end_to_end_correct"])
        self.assertFalse(result["all_output_legs_completed"])
        self.assertEqual(result["public_leg_errors"], response["leg_errors"])

    def test_public_sse_outer_identity_and_bare_cr(self):
        case = self.cases["dev-text-01"]
        request = ns03.default_request(case, session_id="ns03-before-dev-text-01")
        events = [
            {"kind": "vlm_delta", "stage": "vlm", "seq": 0,
             "session_id": request["session_id"], "text": "proof"},
            {"kind": "artifact", "stage": "tts", "seq": 1,
             "session_id": request["session_id"],
             "artifact": {"artifact_id": "audio-0", "encoding": "base64",
                          "data": base64.b64encode(b"\x01\x00").decode(),
                          "index": 0, "total": 1, "bytes": 2,
                          "sha256": ns03.sha256_bytes(b"\x01\x00"),
                          "media_type": "audio", "mime_type": "audio/pcm"}},
            {"kind": "complete", "stage": "vlm", "seq": 2,
             "session_id": request["session_id"], "text": "proof",
             "legs": [{"stage": "tts", "state": "completed"}]},
        ]
        raw = b"".join(b"data: " + json.dumps({"id": "http-outer", "unified": event}).encode() + b"\r\r"
                       for event in events) + b"data: [DONE]\r\r"
        with tempfile.TemporaryDirectory(prefix="ns03-cr-contract-") as tmp:
            result = live.analyze_stream(raw, request, request["session_id"], Path(tmp), "ffmpeg")
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["request_id"], "http-outer")
        self.assertEqual(result["audio"]["artifact_count"], 1)

    def test_artifact_chunks_cannot_change_visual_attribution(self):
        case = self.cases["dev-static-01"]
        request = ns03.default_request(case, session_id="ns03-before-dev-static-01")
        base = {"session_id": request["session_id"]}
        raw_image = b"not-decoded-because-attribution-fails"
        digest = ns03.sha256_bytes(raw_image)
        first = {"artifact_id": "image-0", "encoding": "base64", "index": 0, "total": 2,
                 "data": base64.b64encode(raw_image[:4]).decode(), "media_type": "image",
                 "mime_type": "image/png", "slot_id": "slot-0", "slot_index": 0}
        second = dict(first, index=1, data=base64.b64encode(raw_image[4:]).decode(),
                      slot_id="slot-1", slot_index=1, bytes=len(raw_image), sha256=digest)
        events = [
            dict(base, kind="vlm_delta", stage="vlm", seq=0, text="diagram"),
            dict(base, kind="visual_slot", stage="image", seq=1, request_id="http-image",
                 slot_id="slot-0", slot_index=0, media_type="image", prompt="diagram", text_offset=0),
            dict(base, kind="artifact", stage="image", seq=2, artifact=first),
            dict(base, kind="artifact", stage="image", seq=3, artifact=second),
            dict(base, kind="complete", stage="vlm", seq=4, text="diagram",
                 legs=[{"stage": "image", "slot_id": "slot-0", "state": "completed"}]),
        ]
        raw = b"".join(b"data: " + json.dumps({"id": "http-image", "unified": event}).encode() + b"\n\n"
                       for event in events) + b"data: [DONE]\n\n"
        with tempfile.TemporaryDirectory(prefix="ns03-artifact-contract-") as tmp:
            result = live.analyze_stream(raw, request, request["session_id"], Path(tmp), "ffmpeg")
        self.assertTrue(any("slot_id changed" in error for error in result["errors"]))
        self.assertEqual(len(result["slots"]), 1)
        self.assertTrue(result["parse_ambiguous"])

    def test_owned_process_event_checkpoint_covers_start_and_stop(self):
        events = []
        with tempfile.TemporaryDirectory(prefix="ns03-process-contract-") as tmp:
            process = live.ManagedProcess("owned-true", [sys.executable, "-c", "pass"], Path(tmp) / "owned.log",
                                          None, lambda p, event: events.append((event, p.record())))
            process.process.wait(timeout=5)
            process.stop()
        self.assertEqual([event for event, _ in events], ["start", "stop"])
        self.assertIsNotNone(events[-1][1]["stop_monotonic_ns"])

    @unittest.skipUnless(HAS_OBSERVER_DEPS, "pyzmq and msgpack required for VLM observer tests")
    def test_vlm_observer_preserves_raw_msgpack_and_binds_request(self):
        import msgpack
        import zmq
        with tempfile.TemporaryDirectory(prefix="ns03-vlm-observer-") as tmp:
            root = Path(tmp)
            public_endpoint = f"ipc://{root / 'public'}"
            native_endpoint = f"ipc://{root / 'native'}"
            context = zmq.Context()
            native = context.socket(zmq.PULL)
            native.setsockopt(zmq.RCVTIMEO, 2000)
            native.bind(native_endpoint)
            observer = live.VlmObserver(public_endpoint, native_endpoint)
            sender = context.socket(zmq.PUSH)
            sender.connect(public_endpoint)
            raw = msgpack.packb({"type": "submit", "request_id": "public:vlm",
                                 "data": {"request": {"inputs": {"messages": [
                                     {"role": "system", "content": "director"},
                                     {"role": "user", "content": [{"type": "text", "text": "question"}]},
                                 ]}}}}, use_bin_type=True)
            try:
                case_dir = root / "case"
                case_dir.mkdir()
                observer.begin(case_dir)
                sender.send(raw)
                self.assertEqual(native.recv_multipart(), [raw])
                metadata = observer.finish("public")
                row = json.loads((case_dir / "vlm-prompts.jsonl").read_text().strip())
                self.assertEqual(ns03.sha256_file(case_dir / "vlm-prompts.jsonl"),
                                 metadata["sha256"])
                self.assertEqual(row["raw_base64"], base64.b64encode(raw).decode())
                self.assertEqual(row["messages"], [
                    {"role": "system", "content": "director"},
                    {"role": "user", "content": "question"},
                ])
            finally:
                sender.close(0)
                native.close(0)
                context.term()
                observer.close()

    @unittest.skipUnless(HAS_OBSERVER_DEPS, "pyzmq and msgpack required for VLM capture tests")
    def test_independent_vlm_capture_checks_legacy_and_first_turn_history(self):
        case = self.cases["dev-followup-01"]
        setup = {"clean_text": "exact setup assistant response"}
        current = {"role": "user", "content": case["prompt"]}
        setup_user = {"role": "user", "content": case["setup"]["prompt"]}
        assistant = {"role": "assistant", "content": setup["clean_text"]}
        system = {"role": "system", "content": "director"}
        with tempfile.TemporaryDirectory(prefix="ns03-vlm-verify-") as tmp:
            root = Path(tmp)
            bindings = {}
            for name, messages, expected in (
                    ("setup", [system, setup_user],
                     verify.expected_vlm_messages(case, "setup", None, True)),
                    ("before", [system, current],
                     verify.expected_vlm_messages(case, "answer", setup, False)),
                    ("after", [system, setup_user, assistant, current],
                     verify.expected_vlm_messages(case, "answer", setup, True))):
                case_dir = root / name
                _, reported = write_vlm_capture(case_dir, messages, name + ":vlm")
                metadata = verify.verify_vlm_capture(case_dir, reported, name, expected, bindings)
                self.assertEqual(metadata["native_request_id"], name + ":vlm")
            self.assertEqual(len(bindings), 3)

            bad_dir = root / "bad"
            _, reported = write_vlm_capture(bad_dir, [system, current], "bad:vlm")
            with self.assertRaises(AssertionError):
                verify.verify_vlm_capture(bad_dir, reported, "bad", [setup_user, assistant, current], bindings)

    def test_focused_followup_scorecard_and_independent_history_gate(self):
        runs = {}
        candidate = {"first_turn_history": True}
        for population in ns03.POPULATIONS:
            cases = [case for case in self.suite["cases"]
                     if case["population"] == population and case["class"] == "followup"]
            for arm in ("before", "after"):
                reports = []
                for index, case in enumerate(cases):
                    answer = record()
                    setup = record()
                    answer["setup"] = setup
                    answer["session_history"] = True
                    answer["same_session_as_setup"] = True
                    answer["clean_text"] = (" ".join(group[0] for group in case["context_any"])
                                            if arm == "after" or index == 0 else "unrelated response")
                    answer["case_id"] = case["id"]
                    reports.append(answer)
                runs[f"{population}/{arm}"] = {"case_class": "followup", "cases": reports}
        scorecard = ns03.make_followup_scorecard(
            self.suite, self.prereg, runs, {"runs": "focused-test"}, candidate)
        self.assertEqual(scorecard["format"], ns03.FOLLOWUP_FORMAT)
        self.assertEqual(scorecard["promotion"], "COMPARATOR_REQUIRED")
        self.assertNotIn("thresholds", scorecard)
        self.assertEqual(scorecard["history_gate"], {
            "before": {"cases": 10, "correct": 2, "accuracy": 0.2},
            "after": {"cases": 10, "correct": 10, "accuracy": 1.0},
            "target": 0.8, "before_to_after_gain": 0.8,
            "target_met": True, "verifiable_gain": True,
        })
        self.assertEqual(len(scorecard["misses"]), 8)
        self.assertEqual(scorecard["paired"]["all"]["discordant"], 0)
        before = scorecard["arms"]["development/before"]["cases"] + \
            scorecard["arms"]["heldout/before"]["cases"]
        after = scorecard["arms"]["development/after"]["cases"] + \
            scorecard["arms"]["heldout/after"]["cases"]
        self.assertEqual(verify.followup_history_gate(before, after), scorecard["history_gate"])
        with self.assertRaises(ValueError):
            ns03.make_followup_scorecard(self.suite, self.prereg, runs, {}, {})
        runs["development/before"]["case_class"] = "all"
        with self.assertRaises(ValueError):
            ns03.make_followup_scorecard(self.suite, self.prereg, runs, {}, candidate)

    def test_exact_paired_sign_test(self):
        before = [record() for _ in range(3)]
        after = [record() for _ in range(3)]
        for index in range(3):
            before[index]["case_id"] = after[index]["case_id"] = f"case-{index}"
        # Directly test the frozen discordance calculation with routing fields.
        for item in before:
            item["routing_correct"] = False
        for item in after:
            item["routing_correct"] = True
        before[0]["routing_correct"] = True
        after[0]["routing_correct"] = False
        # Two improvements and one worsening have an exact two-sided p=1.
        self.assertEqual(ns03.exact_mcnemar(before, after),
                         {"cases": 3, "improved": 2, "worsened": 1, "discordant": 3,
                          "two_sided_exact_p": 1.0, "accuracy_delta_percentage_points": 100 / 3})

    def test_scorecard_retains_every_miss_and_hard_ideal_is_separate(self):
        runs = {}
        for population in ns03.POPULATIONS:
            cases = [c for c in self.suite["cases"] if c["population"] == population]
            before_cases = []
            after_cases = []
            for case in cases:
                if case["class"] == "text":
                    before = record(images=1)
                elif case["class"] == "static":
                    before = record(images=1)
                elif case["class"] == "dynamic":
                    before = record()
                else:
                    before = record()
                after = record(images=1 if case["class"] == "static" else 0,
                               videos=1 if case["class"] == "dynamic" else 0)
                before["case_id"] = after["case_id"] = case["id"]
                if case["class"] == "followup":
                    setup = record()
                    before["setup"] = after["setup"] = setup
                    for item in (before, after):
                        item["clean_text"] = " ".join(group[0] for group in case["context_any"])
                        item["session_history"] = True
                        item["same_session_as_setup"] = True
                before_cases.append(before)
                after_cases.append(after)
            runs[f"{population}/before"] = {"cases": before_cases}
            runs[f"{population}/after"] = {"cases": after_cases}
        scorecard = ns03.make_scorecard(self.suite, self.prereg, runs, {}, {"id": "test"})
        self.assertEqual(scorecard["thresholds"]["after_all"]["routing_accuracy"], 1.0)
        self.assertEqual(scorecard["thresholds"]["after_all"]["redundant_visual_rate"], 0.0)
        self.assertTrue(scorecard["thresholds"]["rubric_hard_ideal"]["after_all_met"])
        self.assertEqual(scorecard["paired"]["all"]["improved"], 20)
        self.assertEqual(scorecard["promotion"], "PROMOTION_ELIGIBLE")
        self.assertEqual(len(scorecard["misses"]), 20)


if __name__ == "__main__":
    unittest.main()
