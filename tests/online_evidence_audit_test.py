#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fresh-process independent online-evidence audit and tamper regressions.

The accepted fixture builders run only in the parent test process. Every audit
verdict runs in a new ``python3 -I`` process with producer, scoring, and ML
imports blocked. No GPU, inference, serving, optimizer, or network is used.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "python/umminfer_online_evidence_audit.py"
TASK42 = ROOT / "tests/fixtures/behavior_token_alignment/task42"
TASK42_EVIDENCE = ROOT / "docs/evidence/behavior-token-alignment-real-cpu.json"
PREFLIGHT_EVIDENCE = ROOT / "docs/evidence/online-grpo-preflight-cpu.json"
LEGACY_HASHES = {
    "manifest.json": "4d069bfd013a87640981c887e9b5cec5c25d9acf4776a4c6cac481fefc7c4974",
    "manifest.json.sha256": "d69cbc49efe50ef8fb1190ef16f8bc7cedc2cf0a0a0199e122ac88de64ac4f0c",
    "sft_eval.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "sft_train.jsonl": "95aec6d34f0a27695d15856430299f4b95486ab98cf1e8008eb6e20502b4c3aa",
}

GUARD = r'''
import builtins, runpy, sys
blocked = {
    "umminfer_online_rollout", "umminfer_behavior_token_alignment",
    "umminfer_online_grpo_preflight", "umminfer_sft_dataset",
    "umminfer_ablation_metrics", "umminfer_joint_grpo_live",
    "umminfer_joint_trainer", "torch", "transformers", "numpy", "mlx",
}
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise AssertionError("forbidden audit import: " + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
'''


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def write_json(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(
        encoding="utf-8").splitlines() if line]


def write_jsonl(path, records):
    Path(path).write_bytes(b"".join(canonical(record) + b"\n" for record in records))


def refresh_manifest(bundle, relative):
    bundle = Path(bundle)
    path = bundle / relative
    value = json.loads(path.read_text(encoding="utf-8"))
    trajectory = bundle / value["trajectory"]["path"]
    value["trajectory"].update({
        "sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        "bytes": trajectory.stat().st_size,
    })
    write_json(path, value)
    Path(str(path) + ".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="ascii")


def refresh_preflight_checksums(root):
    root = Path(root)
    path = root / "SHA256SUMS"
    lines = []
    for line in path.read_text(encoding="ascii").splitlines():
        _old, relative = line.split("  ", 1)
        lines.append(hashlib.sha256((root / relative).read_bytes()).hexdigest()
                     + "  " + relative)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def run_audit(arguments, expected=0):
    result = subprocess.run(
        [sys.executable, "-I", "-c", GUARD, str(AUDIT), *map(str, arguments)],
        cwd=ROOT, text=True, capture_output=True, timeout=120)
    if result.returncode != expected:
        raise AssertionError(
            f"audit exit {result.returncode}, expected {expected}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}")
    return result


class OnlineEvidenceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="umm-online-evidence-audit-")
        cls.root = Path(cls.temporary.name)
        sys.path.insert(0, str(ROOT / "tests"))
        import online_grpo_preflight_test as fixture  # fixture generation only
        cls.fixture = fixture
        cls.bundles = {}
        for version in (1, 2):
            work = cls.root / f"v{version}"
            bundle, tokenizer = fixture.make_online_bundle(
                work / "source", version=version, exports=True)
            config = fixture.make_config(bundle, tokenizer)
            output = work / "preflight"
            fixture.preflight.prepare_preflight(
                bundle, output, config, tokenizer_dir=tokenizer)
            cls.bundles[version] = (bundle, tokenizer, output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def preflight_args(self, version=1, evidence=False):
        bundle, tokenizer, output = self.bundles[version]
        args = ["preflight", "--online-pack", bundle,
                "--preflight-bundle", output, "--tokenizer-dir", tokenizer,
                "--repo-root", ROOT]
        if evidence:
            args += ["--accepted-evidence", PREFLIGHT_EVIDENCE]
        return args

    def test_v1_v2_fresh_process_byte_identity_and_checkpoint_identity(self):
        observed_runs = {}
        for version in (1, 2):
            with self.subTest(version=version):
                reports = []
                for index in range(2):
                    path = self.root / f"v{version}-audit-{index}.json"
                    run_audit(self.preflight_args(version) + ["--report", path])
                    reports.append(path.read_bytes())
                self.assertEqual(reports[0], reports[1])
                report = json.loads(reports[0])
                expected_run_id = json.loads(
                    (self.bundles[version][2] / "run_identity.json")
                    .read_text(encoding="utf-8"))["run_id"]
                self.assertEqual(report["status"], "PASS")
                self.assertEqual(report["run_id"], expected_run_id)
                observed_runs[version] = report["run_id"]
                self.assertEqual(report["source_rollout_reconstruction"],
                                 "PASS_BYTE_IDENTICAL")
                self.assertEqual(report["run_identity_reconstruction"],
                                 "PASS_BYTE_IDENTICAL")
                self.assertEqual(report["objective_reconstruction"],
                                 "PASS_BYTE_IDENTICAL")
                self.assertEqual(report["schedule_reconstruction"],
                                 "PASS_BYTE_IDENTICAL")
                self.assertEqual(report["checkpoint_disagreements"], [])
                self.assertEqual(report["groups"][2]["group_std"], 0.0)
        self.assertNotEqual(observed_runs[1], observed_runs[2])

    def test_attested_retry_attempt_is_rebuilt_and_bounded(self):
        work = self.root / "attested-retry"
        bundle, tokenizer = self.fixture.make_online_bundle(
            work / "source", version=2, exports=True, first_attempt=2)
        records = read_jsonl(bundle / "trajectories.jsonl")
        self.assertEqual(records[0]["rollout_provenance"]["attempt"], 2)
        self.assertEqual(records[0]["rollout_provenance"]["response"]["attempt"], "2")
        config = self.fixture.make_config(bundle, tokenizer)
        output = work / "preflight"
        self.fixture.preflight.prepare_preflight(bundle, output, config,
                                                tokenizer_dir=tokenizer)
        report_path = work / "audit.json"
        run_audit(["preflight", "--online-pack", bundle,
                   "--preflight-bundle", output, "--tokenizer-dir", tokenizer,
                   "--repo-root", ROOT, "--report", report_path])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["source_rollout_reconstruction"], "PASS_BYTE_IDENTICAL")

        # A consistently re-attested but out-of-policy attempt is still invalid.
        records[0]["rollout_provenance"]["attempt"] = 5
        response = records[0]["rollout_provenance"]["response"]
        response["attempt"] = "5"
        response["attestation_sha256"] = hashlib.sha256(canonical({
            key: value for key, value in response.items()
            if key != "attestation_sha256"})).hexdigest()
        write_jsonl(bundle / "trajectories.jsonl", records)
        refresh_manifest(bundle, "rollout_manifest.json")
        rejected = work / "rejected.json"
        run_audit(["preflight", "--online-pack", bundle,
                   "--preflight-bundle", output, "--tokenizer-dir", tokenizer,
                   "--repo-root", ROOT, "--report", rejected], expected=2)
        self.assertFalse(rejected.exists())

    def test_large_declared_model_file_uses_streaming_identity(self):
        sys.path.insert(0, str(ROOT / "python"))
        import umminfer_online_evidence_audit as audit
        root = self.root / "large-model-file"
        root.mkdir(exist_ok=True)
        target = root / "model-00001-of-00018.safetensors"
        with target.open("wb") as handle:
            handle.truncate(64 * 1024 * 1024 + 1)
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        self.assertEqual(audit.streaming_file_identity(
            root, target.name), {"path": target.name, "sha256": digest.hexdigest(),
                                 "bytes": target.stat().st_size})

    def test_qwen_added_token_key_binds_terminal_identity(self):
        sys.path.insert(0, str(ROOT / "python"))
        import umminfer_online_evidence_audit as audit
        root = self.root / "qwen-terminal-schema"
        root.mkdir(exist_ok=True)
        write_json(root / "generation_config.json", {"eos_token_id": [248046, 248044]})
        write_json(root / "tokenizer_config.json", {
            "eos_token": "<|im_end|>", "added_tokens_decoder": {
                "248044": {"content": "<|endoftext|>", "special": True},
                "248046": {"content": "<|im_end|>", "special": True},
            }})
        write_json(root / "vocab.json", {})
        write_json(root / "tokenizer.json", {
            "normalizer": {"type": "NFC"}, "decoder": {"type": "ByteLevel"},
            "model": {"type": "BPE"}, "pre_tokenizer": {
                "type": "ByteLevel", "add_prefix_space": False,
                "trim_offsets": False, "use_regex": False}})
        (root / "merges.txt").write_bytes(b"")
        identity = {"available_files": [audit.streaming_file_identity(root, path.name)
                                        for path in sorted(root.iterdir())]}
        _files, _loaded, terminal, _vocab = audit.verify_tokenizer_dir(root, identity, 2)
        self.assertEqual(terminal, [248044, 248046])

    def test_before_evidence_preserves_checkpoint_failure(self):
        report = json.loads((ROOT / "docs/evidence/online-evidence-audit-preflight-cpu.json")
                            .read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(len(report["checkpoint_disagreements"]), 16)
        self.assertTrue(all(
            item["class"] == "CHECKPOINT_GROUP_EXPORT_BINDING"
            for item in report["checkpoint_disagreements"]))
        self.assertTrue(all(
            item["accepted"] == "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
            for item in report["checkpoint_disagreements"]
            if item["path"].endswith("group_exports_sha256")))

    def test_retained_task42_fresh_process_and_accepted_comparison(self):
        outputs = []
        for index in range(2):
            path = self.root / f"task42-{index}.json"
            run_audit(["task42", TASK42, "--accepted-evidence", TASK42_EVIDENCE,
                       "--report", path])
            outputs.append(path.read_bytes())
        self.assertEqual(outputs[0], outputs[1])
        report = json.loads(outputs[0])
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["sampled_token_count"], 61)
        self.assertEqual(report["supervised_token_count"], 49)
        self.assertEqual(report["accepted_cases"][0]["clean_id_relation"], "EXACT")
        self.assertEqual(report["accepted_cases"][1]["clean_id_relation"],
                         "CONTEXT_BOUNDARY_DIFFERENCE_BYTES_EXACT")
        self.assertEqual(report["accepted_cases"][1]["masked_token_count"], 25)
        self.assertEqual(report["accepted_cases"][1]["clean_posthoc_token_count"], 24)
        self.assertEqual(report["tokenizer_reencode"], "NOT_RUN")
        self.assertEqual(len(report["original_checksums"]["retained_verified"]), 20)
        self.assertEqual(len(report["original_checksums"]["not_retained"]), 23)

    def test_accepted_summary_mismatch_is_reported_not_adjusted(self):
        changed = self.root / "accepted-summary.json"
        value = json.loads(PREFLIGHT_EVIDENCE.read_text(encoding="utf-8"))
        value["run_id"] = "0" * 64
        write_json(changed, value)
        report_path = self.root / "accepted-summary-failure.json"
        result = run_audit(["preflight", *self.preflight_args()[1:],
                            "--accepted-evidence", changed,
                            "--report", report_path], expected=1)
        self.assertEqual(result.stdout, "")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertTrue(any(item["path"] == "$['run_id']"
                            and item["accepted"] == "0" * 64
                            for item in report["accepted_evidence_mismatches"]))
        expected_run_id = json.loads(
            (self.bundles[1][2] / "run_identity.json")
            .read_text(encoding="utf-8"))["run_id"]
        self.assertEqual(report["run_id"], expected_run_id)

    def test_source_tamper_classes_fail_without_report_or_ml(self):
        bundle, _tokenizer, output = self.bundles[1]
        source = self.root / "negative-source"
        shutil.copytree(bundle, source)

        def run_mutation(name, mutation, code=None):
            work = self.root / ("negative-" + name)
            if work.exists():
                shutil.rmtree(work)
            shutil.copytree(source, work)
            mutation(work)
            report = self.root / ("negative-report-" + name + ".json")
            args = ["preflight", "--online-pack", work,
                    "--preflight-bundle", output,
                    "--tokenizer-dir", self.bundles[1][1],
                    "--repo-root", ROOT, "--report", report]
            result = run_audit(args, expected=2)
            self.assertEqual(result.stdout, "")
            self.assertFalse(report.exists())
            error = json.loads(result.stderr)
            self.assertEqual(error["status"], "REJECTED")
            if code:
                self.assertEqual(error["code"], code)

        def reordered(work):
            path = work / "trajectories.jsonl"
            rows = read_jsonl(path)
            rows[0], rows[1] = rows[1], rows[0]
            write_jsonl(path, rows)
        run_mutation("reordered", reordered, "STALE_REORDERED_OR_MISSING_ROLLOUT")

        def duplicate(work):
            path = work / "trajectories.jsonl"
            rows = read_jsonl(path)
            rows[1] = copy.deepcopy(rows[0])
            write_jsonl(path, rows)
        run_mutation("duplicate", duplicate, "STALE_REORDERED_OR_MISSING_ROLLOUT")

        def missing(work):
            path = work / "trajectories.jsonl"
            write_jsonl(path, read_jsonl(path)[1:])
        run_mutation("missing", missing, "ROLLOUT_COUNT_MISMATCH")

        def offline(work):
            path = work / "trajectories.jsonl"
            rows = read_jsonl(path)
            rows[0].pop("rollout_provenance")
            write_jsonl(path, rows)
        run_mutation("offline-mix", offline, "OFFLINE_ONLINE_PROVENANCE_MIX")

        def cancelled(work):
            path = work / "trajectories.jsonl"
            rows = read_jsonl(path)
            rows[0]["terminal_state"] = "cancelled"
            write_jsonl(path, rows)
        run_mutation("cancelled", cancelled, "CANCELLED_OR_FAILED_RECORD")

        def nonfinite(work):
            path = work / "trajectories.jsonl"
            path.write_text(path.read_text(encoding="utf-8").replace(
                '"logprobs":[-0.25', '"logprobs":[NaN', 1), encoding="utf-8")
        run_mutation("nonfinite", nonfinite, "NON_FINITE_JSON")

        def raw_tamper(work):
            rows = read_jsonl(work / "trajectories.jsonl")
            path = work / "artifacts" / rows[0]["sse_replay"]["raw_ref"]
            payload = bytearray(path.read_bytes())
            payload[-3] ^= 1
            path.write_bytes(payload)
        run_mutation("raw-cas", raw_tamper, "RAW_CAS_HASH_MISMATCH")

        def model_drift(work):
            path = work / "plan.json"
            plan = json.loads(path.read_text(encoding="utf-8"))
            plan["behavior_policy"]["model_config_sha256"] = "ef" * 32
            write_json(path, plan)
        run_mutation("model-drift", model_drift, "STALE_REORDERED_OR_MISSING_ROLLOUT")

        def cross_group(work):
            path = work / "pack/grpo_train.jsonl"
            rows = read_jsonl(path)
            rows[0]["rollouts"][0]["rollout_provenance"]["group_id"] = "0" * 32
            write_jsonl(path, rows)
        run_mutation("cross-group", cross_group, "RECONSTRUCTED_PACK_MISMATCH")

        def truncated(work):
            path = work / "trajectories.jsonl"
            rows = read_jsonl(path)
            rows[0]["terminal_state"] = "completed"
            raw_name = rows[0]["sse_replay"]["raw_ref"]
            raw_path = work / "artifacts" / raw_name
            raw_path.write_bytes(raw_path.read_bytes().replace(
                b'"finish_reason":"stop"', b'"finish_reason":"length"'))
        run_mutation("truncated", truncated, "RAW_CAS_HASH_MISMATCH")

    def test_not_covered_and_truncated_producers_fail_before_output(self):
        for covered, finish, label in (
                (False, "stop", "not-covered"), (True, "length", "truncated-producer")):
            with self.subTest(label=label):
                work = self.root / label
                bundle, tokenizer = self.fixture.make_online_bundle(
                    work / "source", exports=False, covered=covered,
                    finish_reason=finish)
                report = self.root / (label + ".json")
                result = run_audit([
                    "preflight", "--online-pack", bundle,
                    "--preflight-bundle", self.bundles[1][2],
                    "--tokenizer-dir", tokenizer, "--repo-root", ROOT,
                    "--report", report], expected=2)
                self.assertEqual(result.stdout, "")
                self.assertFalse(report.exists())

    def test_policy_tokenizer_drift_and_consistently_resigned_tamper(self):
        bundle, tokenizer, output = self.bundles[1]
        drifted = self.root / "drifted-preflight"
        shutil.copytree(output, drifted)
        path = drifted / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["current_policy"]["tokenizer_sha256"] = "ef" * 32
        write_json(path, config)
        report = self.root / "tokenizer-drift.json"
        result = run_audit([
            "preflight", "--online-pack", bundle, "--preflight-bundle", drifted,
            "--tokenizer-dir", tokenizer, "--repo-root", ROOT,
            "--report", report], expected=2)
        self.assertEqual(result.stdout, "")
        self.assertFalse(report.exists())

        drifted_tokenizer = self.root / "drifted-tokenizer-files"
        shutil.copytree(tokenizer, drifted_tokenizer)
        with (drifted_tokenizer / "vocab.json").open("ab") as stream:
            stream.write(b" ")
        file_report = self.root / "tokenizer-file-drift.json"
        result = run_audit([
            "preflight", "--online-pack", bundle,
            "--preflight-bundle", output, "--tokenizer-dir", drifted_tokenizer,
            "--repo-root", ROOT, "--report", file_report], expected=2)
        self.assertEqual(result.stdout, "")
        self.assertFalse(file_report.exists())
        self.assertEqual(json.loads(result.stderr)["code"], "TOKENIZER_FILE_DRIFT")

        for label in ("objective-resigned", "schedule-resigned"):
            with self.subTest(label=label):
                tampered = self.root / label
                shutil.copytree(output, tampered)
                name = "objective_fixtures.json" if label.startswith("objective") \
                    else "schedule.json"
                path = tampered / name
                value = json.loads(path.read_text(encoding="utf-8"))
                if name.startswith("objective"):
                    value["groups"][0]["members"][0]["current_policy"]["logprobs"][0] -= 0.01
                else:
                    value["checkpoints"][0]["completed_group_ids"].append("forged")
                write_json(path, value)
                manifest_path = tampered / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["files"][name] = {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                }
                write_json(manifest_path, manifest)
                Path(str(manifest_path) + ".sha256").write_text(
                    hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n",
                    encoding="ascii")
                refresh_preflight_checksums(tampered)
                report = self.root / (label + ".json")
                result = run_audit([
                    "preflight", "--online-pack", bundle, "--preflight-bundle", tampered,
                    "--tokenizer-dir", tokenizer, "--repo-root", ROOT,
                    "--report", report], expected=2)
                self.assertEqual(result.stdout, "")
                self.assertFalse(report.exists())

    def test_task42_tamper_no_output(self):
        for label in ("token-export", "logprob", "raw-sse"):
            with self.subTest(label=label):
                work = self.root / ("task42-" + label)
                shutil.copytree(TASK42, work)
                if label == "token-export":
                    path = work / "aligned/trajectories.jsonl"
                    rows = read_jsonl(path)
                    rows[0]["token_export"]["streams"]["sft_target"]["token_ids"][0] += 1
                    write_jsonl(path, rows)
                elif label == "logprob":
                    path = work / "summary.json"
                    value = json.loads(path.read_text(encoding="utf-8"))
                    value["cases"][0]["behavior"]["logprobs"][0] -= 0.01
                    write_json(path, value)
                else:
                    rows = read_jsonl(work / "source/live/trajectories.jsonl")
                    path = work / "live/artifacts" / rows[0]["sse_replay"]["raw_ref"]
                    payload = bytearray(path.read_bytes())
                    payload[-3] ^= 1
                    path.write_bytes(payload)
                report = self.root / ("task42-" + label + ".json")
                result = run_audit(["task42", work, "--report", report], expected=2)
                self.assertEqual(result.stdout, "")
                self.assertFalse(report.exists())

    def test_repository_isolation_and_default_legacy_pack_hashes(self):
        sys.path.insert(0, str(ROOT / "python"))
        import umminfer_online_evidence_audit as audit
        import umminfer_sft_dataset as dataset  # separate legacy regression only
        isolation = audit.audit_repository_isolation(ROOT)
        self.assertEqual(isolation["status"], "PASS")
        self.assertNotIn("umminfer_online_rollout", isolation["auditor_imports"])
        self.assertNotIn("umminfer_behavior_token_alignment", isolation["auditor_imports"])
        self.assertNotIn("umminfer_online_grpo_preflight", isolation["auditor_imports"])
        output = self.root / "legacy-pack"
        dataset.build_dataset(
            [TASK42 / "source/live/trajectories.jsonl"], output, mode="sft",
            artifact_dir=TASK42 / "live/artifacts", seed="task52-legacy-pin",
            eval_ratio=0.0)
        actual = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in output.iterdir() if path.is_file()}
        self.assertEqual(actual, LEGACY_HASHES)
        for name in ("sft_train.jsonl", "sft_eval.jsonl"):
            for row in read_jsonl(output / name):
                self.assertNotIn("token_export", row)
                self.assertNotIn("rollout_provenance", row)


if __name__ == "__main__":
    unittest.main(verbosity=2)
