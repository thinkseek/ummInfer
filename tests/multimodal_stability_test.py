#!/usr/bin/env python3
"""CPU-only deterministic and fail-closed tests for multimodal stability runs.

All responses are real-shaped synthetic SSE fixtures. No server, model, GPU,
network, resident service, credential, or real evidence is used. Run with the
benchmark environment that supplies pinned jieba 0.42.1, for example:
  /path/to/bench-venv/bin/python -I tests/multimodal_stability_test.py
"""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import multimodal_coverage_audit as coverage  # noqa: E402
import multimodal_stability as stability  # noqa: E402


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class StabilityRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="umm-stability-test-")
        self.root = Path(self.temp.name)
        stability.verify_segmenter()
        self.config, self.manifest, self.baseline = stability.make_replay_workspace(
            self.root / "workspace")

    def tearDown(self):
        self.temp.cleanup()

    def finalize(self, specs=({}, {}, {}), run_completed=True, warmup=True):
        evidence = Path(self.config["evidence_dir"])
        evidence.mkdir(parents=True, exist_ok=True)
        if warmup:
            stability.write_synthetic_round(evidence / "warmup", self.config,
                                            self.manifest)
        for number, options in enumerate(specs, 1):
            stability.write_synthetic_round(evidence / f"round-{number}",
                                            self.config, self.manifest, **options)
        return stability.finalize_evidence(self.config, self.manifest, self.baseline,
                                           replay=True, run_completed=run_completed)

    def assert_reason(self, result, code):
        self.assertIn(code, {reason["code"] for reason in result["reasons"]},
                      result["reasons"])

    def test_three_round_replay_pass_is_not_real_evidence_and_fully_sealed(self):
        result = self.finalize()
        self.assertEqual(result["verdict"], "NOT_REAL_EVIDENCE")
        self.assertEqual(result["replay_stability_verdict"], "STABILITY_PASS")
        self.assertFalse(result["real_stability_pass_claim"])
        self.assertEqual(result["real_execution"], "NOT_RUN")
        self.assertEqual(result["rounds_run"], 3)
        self.assertEqual(result["warmup_rounds"], 1)
        self.assertEqual(result["consecutive_complete_matching_rounds"], 3)
        self.assertEqual(result["reasons"], [])
        evidence = Path(self.config["evidence_dir"])
        self.assertTrue((evidence / "NOT_REAL_EVIDENCE").is_file())
        for number in range(1, 4):
            verification = read_json(evidence / f"round-{number}/verification.json")
            comparison = read_json(evidence / f"round-{number}/comparison.json")
            self.assertTrue(verification["complete"])
            self.assertEqual(verification["case_count"], 70)
            self.assertEqual(verification["raw_valid_case_ids"].__len__(), 70)
            self.assertEqual(verification["missing_metric_case_ids"], [])
            self.assertTrue(verification["zero_valued_metrics"])
            self.assertTrue(all(item["value"] == 0
                                for item in verification["zero_valued_metrics"]))
            self.assertEqual(verification["aggregates"]["completed_terminals"], 70)
            self.assertTrue(comparison["passed"])
            self.assertEqual(len(comparison["cases"]), 70)
            self.assertTrue(all(case["latency"] is not None for case in comparison["cases"]))
        listed = coverage.verify_sha256sums(evidence)
        actual = {path.relative_to(evidence).as_posix()
                  for path in evidence.rglob("*") if path.is_file()
                  and path != evidence / "SHA256SUMS"}
        self.assertEqual(set(listed), actual)
        self.assertIn("round-1/SHA256SUMS", listed)
        self.assertGreater(len(listed), 800)

    def test_schedule_and_raw_replay_are_deterministic(self):
        first = self.finalize()
        first_root = Path(self.config["evidence_dir"])
        other_root = self.root / "second"
        config, manifest, baseline = stability.make_replay_workspace(other_root)
        evidence = Path(config["evidence_dir"])
        evidence.mkdir()
        stability.write_synthetic_round(evidence / "warmup", config, manifest)
        for number in range(1, 4):
            stability.write_synthetic_round(evidence / f"round-{number}", config, manifest)
        second = stability.finalize_evidence(config, manifest, baseline, replay=True)
        self.assertEqual(first["replay_stability_verdict"],
                         second["replay_stability_verdict"])
        self.assertEqual((first_root / "schedule.json").read_bytes(),
                         (evidence / "schedule.json").read_bytes())
        for number in range(1, 4):
            for name in ("cases.jsonl", "verification.json", "comparison.json"):
                self.assertEqual((first_root / f"round-{number}" / name).read_bytes(),
                                 (evidence / f"round-{number}" / name).read_bytes(), name)
        schedule = read_json(first_root / "schedule.json")
        self.assertEqual([step["phase"] for step in schedule["steps"]],
                         ["warmup", "measured", "measured", "measured"])
        self.assertEqual([step["round"] for step in schedule["steps"]], [1, 1, 2, 3])
        manifest_ids = [case["id"] for case in self.manifest["cases"]]
        self.assertTrue(all(step["case_ids"] == manifest_ids
                            for step in schedule["steps"]))

    def test_per_case_and_aggregate_quality_regression_fail(self):
        result = self.finalize([{}, {"wrong_case": "gsm8k-test-0000"}, {}])
        self.assertEqual(result["replay_stability_verdict"], "FAIL")
        self.assertEqual(result["consecutive_complete_matching_rounds"], 1)
        self.assert_reason(result, "QUALITY_REGRESSION")
        comparison = read_json(Path(self.config["evidence_dir"]) /
                               "round-2/comparison.json")
        case = next(item for item in comparison["cases"]
                    if item["case_id"] == "gsm8k-test-0000")
        metric = case["quality"][0]
        self.assertEqual(metric["baseline"], 1.0)
        self.assertEqual(metric["current"], 0.0)
        self.assertEqual(metric["delta"], -1.0)
        self.assertFalse(metric["ok"])
        # The two reference quality misses remain visible and do not fail a
        # matching round merely because they are quality failures.
        round_one = read_json(Path(self.config["evidence_dir"]) /
                              "round-1/comparison.json")
        for case_id in ("gsm8k-test-0008", "gsm8k-test-0012"):
            case = next(item for item in round_one["cases"] if item["case_id"] == case_id)
            self.assertEqual(case["quality"][0]["current"], 0.0)
            self.assertTrue(case["quality"][0]["ok"])

    def test_p95_spike_fails_without_changing_median(self):
        result = self.finalize([{}, {"p95_spike": True}, {}])
        self.assertEqual(result["replay_stability_verdict"], "FAIL")
        self.assert_reason(result, "LATENCY_REGRESSION")
        comparison = read_json(Path(self.config["evidence_dir"]) /
                               "round-2/comparison.json")
        self.assertTrue(comparison["quality_ok"])
        self.assertTrue(comparison["median_ok"])
        self.assertFalse(comparison["p95_ok"])
        p95 = [item for item in comparison["latency"] if item["statistic"] == "p95"]
        self.assertEqual(len(p95), 5)
        self.assertTrue(all(not item["ok"] and item["relative_delta"] > 0.10
                            for item in p95))
        self.assertTrue(all(item["ok"] for item in comparison["latency"]
                            if item["statistic"] == "median"))

    def test_median_and_p95_both_block_latency_drift(self):
        result = self.finalize([{"scale": 1.11}, {}, {}])
        self.assertEqual(result["replay_stability_verdict"], "FAIL")
        comparison = read_json(Path(self.config["evidence_dir"]) /
                               "round-1/comparison.json")
        self.assertFalse(comparison["median_ok"])
        self.assertFalse(comparison["p95_ok"])
        self.assertTrue(all(not item["ok"] for item in comparison["latency"]))

    def test_failed_case_is_missing_not_zero_and_fails_round(self):
        result = self.finalize([{}, {"failed_case": "gsm8k-test-0000"}, {}])
        self.assertEqual(result["replay_stability_verdict"], "FAIL")
        self.assert_reason(result, "ROUND_CONTRACT_FAILED")
        comparison = read_json(Path(self.config["evidence_dir"]) /
                               "round-2/comparison.json")
        case = next(item for item in comparison["cases"]
                    if item["case_id"] == "gsm8k-test-0000")
        self.assertEqual(case["coverage"], "MISSING")
        self.assertIsNone(case["quality"][0]["current"])
        self.assertEqual(case["quality"][0]["coverage"], "MISSING")
        self.assertNotEqual(case["quality"][0]["current"], 0)
        verification = read_json(Path(self.config["evidence_dir"]) /
                                 "round-2/verification.json")
        self.assertFalse(verification["complete"])
        self.assertTrue(any(failure.get("category") == "infrastructure"
                            for failure in verification["hard_failures"]))

    def test_partial_round_cannot_be_completed_from_other_rounds(self):
        omitted = "vqa2-val-262148000"
        result = self.finalize([{}, {"omit_case": omitted}, {}])
        self.assertEqual(result["replay_stability_verdict"], "FAIL")
        self.assert_reason(result, "PARTIAL_ROUND")
        self.assert_reason(result, "MISSING_METRIC")
        comparison = read_json(Path(self.config["evidence_dir"]) /
                               "round-2/comparison.json")
        self.assertEqual(comparison["missing_case_ids"], [omitted])
        self.assertFalse(comparison["coverage_complete"])
        case = next(item for item in comparison["cases"] if item["case_id"] == omitted)
        self.assertIsNone(case["quality"][0]["current"])
        self.assertEqual(case["coverage"], "MISSING")

    def test_missing_raw_event_file_is_a_failed_partial_round(self):
        evidence = Path(self.config["evidence_dir"])
        evidence.mkdir(parents=True, exist_ok=True)
        for number in range(1, 4):
            stability.write_synthetic_round(evidence / f"round-{number}",
                                            self.config, self.manifest)
        case_id = self.manifest["cases"][0]["id"]
        (evidence / f"round-2/events/{case_id}.jsonl").unlink()
        result = stability.finalize_evidence(self.config, self.manifest, self.baseline,
                                             replay=True)
        self.assertEqual(result["replay_stability_verdict"], "FAIL")
        verification = read_json(evidence / "round-2/verification.json")
        self.assertFalse(verification["complete"])
        self.assertTrue(any(error["case_id"] == case_id
                            for error in verification["raw_errors"]))

    def test_live_missing_baseline_writes_machine_readable_rejection(self):
        raw = read_json(self.config["declared_config_path"])
        raw["source"]["require_clean"] = True
        raw["evidence_dir"] = str(self.root / "live-missing-baseline")
        raw["baseline"]["path"] = str(self.root / "absent-baseline.json")
        config_path = self.root / "live-missing-baseline.json"
        stability.write_json(config_path, raw)
        result = stability.run_live(config_path)
        self.assertEqual(result["verdict"], "BASELINE_REQUIRED")
        self.assertEqual(result["rounds_run"], 0)
        self.assert_reason(result, "BASELINE_MISSING")
        evidence = Path(raw["evidence_dir"])
        self.assertFalse((evidence / "round-1").exists())
        self.assertEqual(read_json(evidence / "summary.json")["reasons"][0]["code"],
                         "BASELINE_MISSING")
        coverage.verify_sha256sums(evidence)

    def test_two_clean_rounds_remain_pending(self):
        result = self.finalize(specs=({}, {}), run_completed=False)
        self.assertEqual(result["verdict"], "NOT_REAL_EVIDENCE")
        self.assertEqual(result["replay_stability_verdict"], "STABILITY_PENDING")
        self.assertEqual(result["rounds_run"], 2)
        self.assertEqual(result["consecutive_complete_matching_rounds"], 2)
        self.assert_reason(result, "MEASURED_ROUNDS_PENDING")

    def test_missing_corrupt_synthetic_or_unsafe_baselines_are_rejected(self):
        missing = copy.deepcopy(self.config)
        missing["baseline"]["path"] = str(self.root / "does-not-exist.json")
        with self.assertRaises(stability.StabilityError) as caught:
            stability.load_baseline(missing, self.manifest)
        self.assertEqual(caught.exception.reason["code"], "BASELINE_MISSING")
        with self.assertRaises(stability.StabilityError) as caught:
            stability.load_baseline(self.config, self.manifest, replay=False)
        self.assertEqual(caught.exception.reason["code"], "BASELINE_NOT_REAL")

        baseline_path = Path(self.config["baseline"]["path"])
        original = baseline_path.read_bytes()
        baseline_path.write_bytes(original + b"\n")
        with self.assertRaises(stability.StabilityError) as caught:
            stability.load_baseline(self.config, self.manifest, replay=True)
        self.assertEqual(caught.exception.reason["code"], "BASELINE_SHA256_MISMATCH")
        baseline_path.write_bytes(original)

        for label, mutate, expected in (
                ("unsafe-pass", lambda doc: doc.update(
                    real_model_evidence=True, evidence_class="real", real_cases=70,
                    unique_real_cases=70, http_200=70,
                    stability_verdict="STABILITY_PASS", stability_pass_claim=True),
                 "BASELINE_UNSAFE_CLAIM"),
                ("incomplete", lambda doc: doc.update(
                    case_results=doc["case_results"][:-1]), "BASELINE_INCOMPLETE"),
                ("missing-metric", lambda doc: doc["case_results"][0]["metric"].update(
                    value=None), "BASELINE_METRICS_MISSING")):
            with self.subTest(label=label):
                document = read_json(baseline_path)
                mutate(document)
                stability.write_json(baseline_path, document)
                config = copy.deepcopy(self.config)
                config["baseline"]["sha256"] = stability.sha256_file(baseline_path)
                with self.assertRaises(stability.StabilityError) as caught:
                    stability.load_baseline(config, self.manifest,
                                            replay=(label != "unsafe-pass"))
                self.assertEqual(caught.exception.reason["code"], expected)
                baseline_path.write_bytes(original)

    def test_boolean_flip_cannot_promote_synthetic_baseline(self):
        path = Path(self.config["baseline"]["path"])
        document = read_json(path)
        document["real_model_evidence"] = True
        stability.write_json(path, document)
        config = copy.deepcopy(self.config)
        config["baseline"]["sha256"] = stability.sha256_file(path)
        with self.assertRaises(stability.StabilityError) as caught:
            stability.load_baseline(config, self.manifest, replay=False)
        self.assertEqual(caught.exception.reason["code"], "BASELINE_NOT_REAL")
        document["evidence_class"] = "real"
        stability.write_json(path, document)
        config["baseline"]["sha256"] = stability.sha256_file(path)
        with self.assertRaises(stability.StabilityError) as caught:
            stability.load_baseline(config, self.manifest, replay=False)
        self.assertEqual(caught.exception.reason["code"], "BASELINE_NOT_REAL")

    def test_malformed_baseline_report_rows_and_latency_fail_without_crash(self):
        array_path = Path(self.config["baseline"]["path"])
        original = array_path.read_bytes()
        stability.write_json(array_path, [])
        config = copy.deepcopy(self.config)
        config["baseline"]["sha256"] = stability.sha256_file(array_path)
        with self.assertRaises(stability.StabilityError) as caught:
            stability.load_baseline(config, self.manifest, replay=True)
        self.assertEqual(caught.exception.reason["code"], "BASELINE_UNSAFE_SCHEMA")
        array_path.write_bytes(original)

        for label in ("report-array", "row-not-object", "latency-null"):
            with self.subTest(label=label):
                config, manifest, baseline = stability.make_replay_workspace(
                    self.root / label)
                evidence = Path(config["evidence_dir"])
                evidence.mkdir()
                stability.write_synthetic_round(evidence / "warmup", config, manifest)
                for number in range(1, 4):
                    stability.write_synthetic_round(evidence / f"round-{number}",
                                                    config, manifest)
                report_path = evidence / "round-2/report.json"
                cases_path = evidence / "round-2/cases.jsonl"
                if label == "report-array":
                    stability.write_json(report_path, [])
                elif label == "row-not-object":
                    lines = cases_path.read_text().splitlines()
                    lines[0] = "[]"
                    cases_path.write_text("\n".join(lines) + "\n")
                else:
                    report = read_json(report_path)
                    report["cases"][0]["latency"] = None
                    stability.write_json(report_path, report)
                    rows = [json.loads(line) for line in cases_path.read_text().splitlines()]
                    rows[0]["latency"] = None
                    stability.write_jsonl(cases_path, rows)
                result = stability.finalize_evidence(config, manifest, baseline, replay=True)
                self.assertEqual(result["replay_stability_verdict"], "FAIL")
                self.assertFalse(read_json(evidence /
                                           "round-2/verification.json")["complete"])

    def test_warmup_is_required_and_request_tampering_is_rejected(self):
        missing_warmup = self.finalize(warmup=False)
        self.assertEqual(missing_warmup["replay_stability_verdict"], "FAIL")
        self.assertEqual(missing_warmup["consecutive_complete_matching_rounds"], 3)
        self.assert_reason(missing_warmup, "WARMUP_MISSING")

        config, manifest, baseline = stability.make_replay_workspace(
            self.root / "request-tampering")
        evidence = Path(config["evidence_dir"])
        evidence.mkdir()
        stability.write_synthetic_round(evidence / "warmup", config, manifest)
        for number in range(1, 4):
            stability.write_synthetic_round(evidence / f"round-{number}", config, manifest)
        report_path = evidence / "round-2/report.json"
        cases_path = evidence / "round-2/cases.jsonl"
        report = read_json(report_path)
        report["config"]["temperature"] = 0.1
        report["cases"][0]["request"]["seed"] = 42
        stability.write_json(report_path, report)
        rows = [json.loads(line) for line in cases_path.read_text().splitlines()]
        rows[0]["request"]["seed"] = 42
        stability.write_jsonl(cases_path, rows)
        result = stability.finalize_evidence(config, manifest, baseline, replay=True)
        self.assertEqual(result["replay_stability_verdict"], "FAIL")
        verification = read_json(evidence / "round-2/verification.json")
        self.assertFalse(verification["request_config_match"])
        self.assertFalse(verification["request_rows_match"])

    def test_preflight_has_no_real_execution_or_evidence_claim(self):
        config = copy.deepcopy(self.config)
        config["evidence_dir"] = str(self.root / "preflight-result")
        result = stability.finalize_evidence(config, self.manifest, self.baseline,
                                             replay=False, run_completed=False,
                                             executed=False)
        self.assertEqual(result["verdict"], "STABILITY_PENDING")
        self.assertFalse(result["real_model_evidence"])
        self.assertEqual(result["real_execution"], "NOT_RUN")
        self.assertFalse(result["real_stability_pass_claim"])

    def test_changed_binary_model_manifest_data_config_and_source_are_rejected(self):
        provenance = stability.verify_artifacts(self.config, replay=True)
        self.assertEqual(set(provenance["binaries"]), {"gateway", "asr", "vlm"})
        checks = []

        def expect(code, path, payload=b"changed\n"):
            original = Path(path).read_bytes()
            Path(path).write_bytes(original + payload)
            try:
                with self.assertRaises(stability.StabilityError) as caught:
                    stability.verify_artifacts(self.config, replay=True)
                checks.append((code, caught.exception.reason["code"]))
            finally:
                Path(path).write_bytes(original)

        expect("BINARY_CHANGED", self.config["binaries"]["gateway"]["path"])
        asr = self.config["models"]["asr"]
        expect("MODEL_CONFIG_CHANGED", Path(asr["root"]) / "config.json")
        expect("MODEL_WEIGHT_CHANGED", Path(asr["root"]) / "model.safetensors")
        expect("MODEL_WEIGHT_MANIFEST_CHANGED", asr["weights_manifest"]["path"])
        extra_weight = Path(asr["root"]) / "unlisted.safetensors"
        extra_weight.write_bytes(b"unlisted synthetic weight\n")
        try:
            with self.assertRaises(stability.StabilityError) as caught:
                stability.verify_artifacts(self.config, replay=True)
            self.assertEqual(caught.exception.reason["code"], "MODEL_WEIGHT_SET_CHANGED")
        finally:
            extra_weight.unlink()
        expect("RUN_CONFIG_CHANGED", self.config["declared_config_path"])
        self.assertEqual(checks, [(expected, expected) for expected, _ in checks])

        changed = copy.deepcopy(self.config)
        changed["manifest"]["sha256"] = "0" * 64
        with self.assertRaises(stability.StabilityError) as caught:
            stability.verify_artifacts(changed, replay=True)
        self.assertEqual(caught.exception.reason["code"], "MANIFEST_CHANGED")

        actual = copy.deepcopy(self.config["data"]["case_input_sha256"])
        actual[self.manifest["cases"][0]["id"]] = "0" * 64
        with self.assertRaises(stability.StabilityError) as caught:
            stability.check_data_identities(self.config, actual)
        self.assertEqual(caught.exception.reason["code"], "DATA_CHANGED")

        with self.assertRaises(stability.StabilityError) as caught:
            stability.verify_source(self.config)
        self.assertEqual(caught.exception.reason["code"], "SOURCE_CHANGED")

    def test_declared_schedule_sampling_and_isolated_topology_fail_closed(self):
        original_path = Path(self.config["declared_config_path"])
        raw = read_json(original_path)
        variants = (
            ("rounds", lambda doc: doc["schedule"].update(measured_rounds=2),
             "INVALID_SCHEDULE"),
            ("warmup", lambda doc: doc["schedule"].update(warmup_rounds=0),
             "INVALID_SCHEDULE"),
            ("temperature", lambda doc: doc["request"].update(temperature=0.1),
             "INVALID_CONFIG"),
            ("asr-route", lambda doc: doc["request"].update(asr_route="audio"),
             "INVALID_CONFIG"),
            ("resident-port", lambda doc: doc["topology"].update(
                base_url="http://127.0.0.1:43301"), "UNSAFE_TOPOLOGY"),
            ("foreign-gpu", lambda doc: doc["topology"]["gpu_by_stage"].update(vlm=6),
             "UNSAFE_TOPOLOGY"),
            ("shared-gpu", lambda doc: doc["topology"]["gpu_by_stage"].update(vlm=1),
             "UNSAFE_TOPOLOGY"),
            ("non-loopback", lambda doc: doc["topology"].update(
                base_url="http://192.0.2.1:43999"), "UNSAFE_TOPOLOGY"),
        )
        for label, mutate, expected in variants:
            with self.subTest(label=label):
                document = copy.deepcopy(raw)
                mutate(document)
                path = self.root / f"invalid-{label}.json"
                stability.write_json(path, document)
                with self.assertRaises(stability.StabilityError) as caught:
                    stability.normalize_config(document, path, replay=True)
                self.assertEqual(caught.exception.reason["code"], expected)

    def test_self_check_cli_shape_is_replay_only(self):
        output = self.root / "cli-replay"
        result = stability.run_replay_self_check(output)
        self.assertEqual(result["verdict"], "NOT_REAL_EVIDENCE")
        self.assertEqual(result["replay_stability_verdict"], "STABILITY_PASS")
        self.assertFalse(result["real_stability_pass_claim"])
        self.assertTrue((output / "evidence/NOT_REAL_EVIDENCE").is_file())
        # No live evidence directory can be created from the synthetic baseline.
        with self.assertRaises(stability.StabilityError):
            stability.load_baseline(self.config, self.manifest, replay=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
