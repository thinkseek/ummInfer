#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only sampled-behavior/token_export alignment and tamper regressions."""

import argparse
import builtins
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import narrative_rl_fixtures as fixture  # noqa: E402
import umminfer_behavior_token_alignment as align  # noqa: E402
import umminfer_online_rollout as online  # noqa: E402
import umminfer_rl_trajectory as traj  # noqa: E402
import umminfer_sse_replay as replay  # noqa: E402
import umminfer_token_export as tokens  # noqa: E402

REAL_FIXTURE = ROOT / "tests/fixtures/behavior_token_alignment/task42"
REAL_REPORT = ROOT / "docs/evidence/behavior-token-alignment-real-cpu.json"


def _parse_real_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--real-tokenizer-dir")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


REAL_ARGS = _parse_real_args()


def canonical(value):
    return tokens._canonical(value)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


class V1Tokenizer:
    def __init__(self):
        self.special = tuple(sorted((
            "[[image: ", "[[video: ", " | source=slot-0]]", "]]",
            "<spoken>", "</spoken>", "\n\n", ", a",
        ), key=lambda value: (-len(value), value)))
        self.special_ids = {piece: 0x110001 + index
                            for index, piece in enumerate(self.special)}
        self.id_pieces = {value: key for key, value in self.special_ids.items()}
        self.unk_token_id = 0

    def __len__(self):
        return 0x110001 + len(self.special)

    def __call__(self, text, **_kwargs):
        ids, offsets, cursor = [], [], 0
        while cursor < len(text):
            piece = next((candidate for candidate in self.special
                          if text.startswith(candidate, cursor)), text[cursor])
            ids.append(self.special_ids[piece] if piece in self.special_ids
                       else ord(piece) + 1)
            offsets.append((cursor, cursor + len(piece)))
            cursor += len(piece)
        return {"input_ids": ids, "offset_mapping": offsets}

    def decode(self, ids, **_kwargs):
        return "".join(self.id_pieces[value] if value in self.id_pieces
                       else chr(value - 1) for value in ids)


class V2Tokenizer:
    class Normalizer:
        @staticmethod
        def normalize_str(text):
            import unicodedata
            return unicodedata.normalize("NFC", text)

    class Backend:
        normalizer = None

        def id_to_token(self, token_id):
            if not 1 <= token_id <= 256:
                return None
            inverse = {value: key for key, value in tokens._BYTELEVEL_INVERSE.items()}
            return inverse[token_id - 1]

    def __init__(self):
        self.backend_tokenizer = self.Backend()
        self.backend_tokenizer.normalizer = self.Normalizer()
        self.unk_token_id = None

    def __len__(self):
        return 257

    def __call__(self, text, **_kwargs):
        units, normalized = tokens._nfc_source_units(text)
        raw = normalized.encode("utf-8")
        offsets = []
        for cursor in range(len(raw)):
            unit = next(unit for unit in units
                        if unit["normalized_start"] <= cursor < unit["normalized_end"])
            offsets.append((unit["char_start"], unit["char_end"]))
        return {"input_ids": [value + 1 for value in raw], "offset_mapping": offsets}

    def decode(self, ids, **_kwargs):
        return bytes(value - 1 for value in ids).decode("utf-8")


def tokenizer_directory(root, version):
    path = root / "tokenizer"
    path.mkdir(parents=True)
    model = ({"format": "deterministic-v1"} if version == 1 else {
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False,
                          "trim_offsets": False, "use_regex": False},
        "decoder": {"type": "ByteLevel"}, "model": {"type": "BPE"},
    })
    write_json(path / "tokenizer.json", model)
    write_json(path / "config.json", {"fixture": version})
    (path / "merges.txt").write_text("#version: 0.2\n", encoding="ascii")
    write_json(path / "vocab.json", {"<eos>": 0})
    write_json(path / "generation_config.json", {"eos_token_id": [0]})
    write_json(path / "tokenizer_config.json", {
        "eos_token": "<eos>", "added_tokens_decoder": {
            "0": {"content": "<eos>", "special": True},
        },
    })
    return path


def multilingual_events():
    first = "<spoken>黎明🌅 é"
    second = " 字。</spoken> 尾🧪"
    clean = first + second
    return [
        fixture.envelope(0, "vlm_delta", first),
        fixture.visual_slot(1, 0, "image", "灯笼 🏮 é", len(first.encode("utf-8"))),
        fixture.envelope(2, "vlm_delta", second),
        fixture.envelope(3, "complete", clean, legs=[{
            "stage": "image", "slot_id": "slot-0", "slot_index": 0,
            "state": "completed",
        }]),
    ]


def raw_with_behavior(events, evidence, finish_reason="stop"):
    frames = fixture.sse_bytes(events).split(b"\n\n")
    for index in range(len(frames) - 1, -1, -1):
        if frames[index].startswith(b"data: ") and frames[index] != b"data: [DONE]":
            chunk = json.loads(frames[index][6:])
            chunk["choices"][0]["finish_reason"] = finish_reason
            if evidence is not None:
                chunk["behavior"] = copy.deepcopy(evidence)
            frames[index] = b"data: " + canonical(chunk)
            break
    return b"\n\n".join(frames)


def make_record(root, body, events, evidence, finish_reason="stop"):
    recorder = traj.TrajectoryRecorder(root / "live/trajectories.jsonl")
    raw = raw_with_behavior(events, evidence, finish_reason)
    capture = traj.capture_sse_response(io.BytesIO(raw), clock=replay.LogicalTickClock())
    details = {}
    request_id, normalized, output, state, error = traj.parse_sse_stream(
        io.BytesIO(capture.raw_bytes), artifact_dir=recorder.artifact_dir,
        details=details, capture=capture)
    params, option_inputs = traj.request_parameter_references(body, None)
    inputs = traj.input_references(body["messages"], None) + option_inputs
    record = recorder.record(
        session_id=body["session_id"], request_id=request_id,
        endpoint="/v1/chat/completions", inputs=inputs, params=params,
        events=normalized, terminal_state=state, error=error, output_text=output,
        artifacts=details["artifacts"], capture=capture)
    return record, raw


def expected_mask(record):
    stream = record["token_export"]["streams"]["sft_target"]
    result = [1] * stream["token_count"]
    for span in stream["spans"]:
        if span["span_type"] in align.VISUAL_SPANS:
            for index in range(span["token_start"], span["token_end"]):
                result[index] = 0
    return result


def make_bundle(root, version, events=None, finish_reason="stop"):
    root = Path(root)
    tokenizer_dir = tokenizer_directory(root, version)
    tokenizer = V1Tokenizer() if version == 1 else V2Tokenizer()
    events = copy.deepcopy(events if events is not None else fixture.linked_events())
    body = {
        "model": "narrative-fixture", "stream": True,
        "session_id": events[0]["session_id"],
        "messages": [{"role": "user", "content": "tell a multilingual visual story"}],
        "max_tokens": 96, "temperature": 0.0, "top_p": 1.0, "seed": 42,
        "behavior": {"logprobs": True},
    }
    base, _ = make_record(root, body, events, None)
    kwargs = dict(
        tokenizer_dir=tokenizer_dir, tokenizer_id=f"fixture-v{version}",
        tokenizer_revision="fixture-revision", tokenization_version=f"fixture-tokenization-v{version}",
        format_version=version)
    base_export = tokens.attach_token_export(base, tokenizer, **kwargs)
    digest = align._loaded_tokenizer_digest(tokenizer_dir)
    target = base_export["token_export"]["streams"]["sft_target"]
    evidence = {
        "format": align.BEHAVIOR_FORMAT, "tokenizer_sha256": digest,
        "token_ids": target["token_ids"],
        "logprobs": [-(index + 1) / 8.0 for index in range(target["token_count"])],
        "mask": expected_mask(base_export),
    }
    source, raw = make_record(root, body, events, evidence, finish_reason)
    aligned = tokens.attach_token_export(source, tokenizer, **kwargs)
    name = "fixture"
    case = {"name": name, "record_index": 0}
    write_json(root / "source/live/trajectories.jsonl", source)
    # JSONL, not a JSON document: retain exactly one canonical record plus newline.
    (root / "source/live/trajectories.jsonl").write_bytes(canonical(source) + b"\n")
    write_json(root / "aligned/trajectories.jsonl", aligned)
    (root / "aligned/trajectories.jsonl").write_bytes(canonical(aligned) + b"\n")
    (root / "live/trajectories.jsonl").unlink()
    write_json(root / f"requests/{name}.json", body)
    target_text = tokens.reconstruct_sft_target_segments(aligned)[0]
    write_json(root / f"logs/{name}-replay.stdout", {
        "behavior": evidence, "raw_text": target_text,
        "clean_text": source["clean_text"], "finish_reason": "stop",
        "prompt_token_ids": [7, 8],
    })
    provenance = {
        "revision": "UNKNOWN", "model_sha256": "11" * 32,
        "config_sha256": "22" * 32, "tokenizer_sha256": digest,
        "source_commit": "deterministic-contract-fixture",
    }
    summary_stub = {"provenance": provenance}
    manifest_stub = {
        "batch_id": f"behavior-fixture-v{version}",
        "behavior_policy": {"policy_id": "fixture-policy", "revision": "fixture-v1"},
    }
    plan, item = align._plan_and_item(manifest_stub, summary_stub, body, case)
    attestation = online._expected_attestation(
        item, plan, http_status=200, behavior_evidence="exposed", attempt=0)
    headers = {header: str(attestation[key]) for key, header in online.RESPONSE_HEADERS.items()}
    headers.update({"content-type": "text/event-stream", "content-length": str(len(raw))})
    write_json(root / f"live/responses/{name}.json", {"status": 200, "headers": headers})
    summary = {
        "schema": "umminfer-behavior-logprobs-live-evidence-v1", "status": "PASS",
        "provenance": provenance,
        "cases": [{
            "case": name, "coverage": "exposed", "terminal_state": source["terminal_state"],
            "raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_bytes": len(raw),
            "attestation_sha256": attestation["attestation_sha256"],
            "behavior": align._behavior_summary(evidence),
        }],
    }
    write_json(root / "summary.json", summary)
    identity = aligned["token_export"]["tokenizer"]
    source_files = {
        "live/trajectories.jsonl": root / "source/live/trajectories.jsonl",
        "summary.json": root / "summary.json",
        f"requests/{name}.json": root / f"requests/{name}.json",
        f"live/responses/{name}.json": root / f"live/responses/{name}.json",
        f"logs/{name}-replay.stdout": root / f"logs/{name}-replay.stdout",
    }
    for item in source["artifacts"]:
        if item.get("ref"):
            source_files[f"live/artifacts/{item['ref']}"] = root / "live/artifacts" / item["ref"]
    source_files[f"live/artifacts/{source['sse_replay']['raw_ref']}"] = (
        root / "live/artifacts" / source["sse_replay"]["raw_ref"])
    checksum_text = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}\n"
        for relative, path in sorted(source_files.items()))
    (root / "SHA256SUMS.source").write_text(checksum_text, encoding="ascii")
    manifest = {
        "schema": align.ALIGNMENT_SCHEMA, "evidence_class": "DETERMINISTIC_CONTRACT_FIXTURE",
        "batch_id": manifest_stub["batch_id"], "behavior_policy": manifest_stub["behavior_policy"],
        "source": {
            "trajectory": "source/live/trajectories.jsonl",
            "trajectory_sha256": hashlib.sha256(canonical(source) + b"\n").hexdigest(),
            "aligned_trajectory": "aligned/trajectories.jsonl",
            "aligned_trajectory_sha256": hashlib.sha256(canonical(aligned) + b"\n").hexdigest(),
            "artifact_dir": "live/artifacts", "summary": "summary.json",
            "summary_sha256": hashlib.sha256((root / "summary.json").read_bytes()).hexdigest(),
            "checksums": "SHA256SUMS.source",
            "checksums_sha256": hashlib.sha256(checksum_text.encode("ascii")).hexdigest(),
        },
        "tokenizer": {
            "identity": {key: identity[key] for key in ("id", "revision", "tokenization_version")},
            "behavior_sha256": digest, "files": identity["available_files"],
            "terminal_token_ids": [0],
        },
        "cases": [{
            **case,
            "request_file_sha256": hashlib.sha256((root / "requests/fixture.json").read_bytes()).hexdigest(),
            "response_file_sha256": hashlib.sha256((root / "live/responses/fixture.json").read_bytes()).hexdigest(),
            "replay_file_sha256": hashlib.sha256((root / "logs/fixture-replay.stdout").read_bytes()).hexdigest(),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "evidence_sha256": hashlib.sha256(canonical(evidence)).hexdigest(),
            "attestation_sha256": attestation["attestation_sha256"],
        }],
        "excluded_cases": [],
    }
    write_json(root / "alignment.json", manifest)
    return root, tokenizer_dir, tokenizer, source, aligned, evidence


def read_record(path):
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def update_jsonl(path, records):
    path.write_bytes(b"".join(canonical(record) + b"\n" for record in records))


def refresh_file_hash(root, key):
    manifest_path = root / "alignment.json"
    manifest = json.loads(manifest_path.read_text())
    relative = manifest["source"][key]
    hash_key = key + "_sha256"
    manifest["source"][hash_key] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    write_json(manifest_path, manifest)


class BehaviorTokenAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_tmp = tempfile.TemporaryDirectory(prefix="umm-behavior-align-fixtures-")
        base = Path(cls.base_tmp.name)
        cls.v1 = make_bundle(base / "v1", 1)
        cls.v2 = make_bundle(base / "v2", 2, multilingual_events())

    @classmethod
    def tearDownClass(cls):
        cls.base_tmp.cleanup()

    def copied(self, version):
        temporary = tempfile.TemporaryDirectory(prefix=f"umm-behavior-align-v{version}-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / "bundle"
        shutil.copytree(getattr(self, f"v{version}")[0], root)
        return root, root / "tokenizer"

    def test_valid_v1_v2_strong_and_fresh_process_cli(self):
        for version in (1, 2):
            with self.subTest(version=version):
                root, tokenizer_dir, tokenizer, _, _, _ = getattr(self, f"v{version}")
                report = align.validate_bundle(root, tokenizer_dir, tokenizer=tokenizer)
                self.assertEqual(report["status"], "PASS")
                self.assertEqual(report["validation_level"], "STRONG")
                self.assertTrue(report["accepted_cases"][0]["sampled_posthoc_target_ids_equal"])
                self.assertEqual(report["raw_sse_fresh_replay"], "PASS_BYTE_IDENTICAL")
                self.assertNotIn("torch", sys.modules)
                self.assertNotIn("transformers", sys.modules)
                with tempfile.TemporaryDirectory(prefix="umm-behavior-align-cli-") as tmp:
                    output = Path(tmp) / "report.json"
                    proc = subprocess.run([
                        sys.executable, "-I", str(ROOT / "python/umminfer_behavior_token_alignment.py"),
                        str(root), "--tokenizer-dir", str(tokenizer_dir), "--report", str(output),
                    ], capture_output=True, text=True, timeout=120, check=False)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    cli = json.loads(output.read_text())
                    self.assertEqual(cli["status"], "PASS")
                    self.assertEqual(cli["tokenizer_reencode"], "NOT_RUN")
                    self.assertEqual(json.loads(proc.stdout), cli)

    def test_retained_real_61_tokens_and_report_regression(self):
        summary = json.loads((REAL_FIXTURE / "summary.json").read_text())
        aligned = [json.loads(line) for line in
                   (REAL_FIXTURE / "aligned/trajectories.jsonl").read_text().splitlines() if line]
        report = json.loads(REAL_REPORT.read_text())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["validation_level"], "STRONG")
        self.assertEqual(sum(case["sampled_token_count"] for case in report["accepted_cases"]), 61)
        self.assertEqual(sum(case["masked_token_count"] for case in report["accepted_cases"]), 49)
        self.assertTrue(report["accepted_cases"][0]["clean_reencoded_ids_equal"])
        self.assertFalse(report["accepted_cases"][1]["clean_reencoded_ids_equal"])
        self.assertEqual(report["accepted_cases"][1]["clean_id_relation"],
                         "CONTEXT_BOUNDARY_DIFFERENCE_BYTES_EXACT")
        for index, case in enumerate(report["accepted_cases"]):
            record = aligned[index]
            result = tokens.validate_record_token_export(record)
            self.assertEqual(result["status"], "PASS")
            behavior = summary["cases"][index]["behavior"]
            self.assertEqual(record["token_export"]["streams"]["sft_target"]["token_ids"],
                             behavior["token_ids"])
            self.assertEqual(case["export_sha256"], record["token_export"]["export_sha256"])
        self.assertEqual([case["name"] for case in report["excluded_cases"]],
                         ["truncated", "cancel", "default"])

    def test_sampled_target_reorder_cross_record_and_eos_reject(self):
        _, _, _, _, aligned, evidence = self.v1
        terminal = [0]
        for name, mutate in (
            ("target-mismatch", lambda value: value["token_ids"].__setitem__(0, 999999)),
            ("reordered", lambda value: value["token_ids"].reverse()),
            ("cross-record", lambda value: value.__setitem__(
                "token_ids", [value["token_ids"][-1]] + value["token_ids"][:-1])),
        ):
            with self.subTest(name=name):
                bad = copy.deepcopy(evidence)
                mutate(bad)
                with self.assertRaisesRegex(ValueError, "sampled ids differ"):
                    align._validate_alignment(copy.deepcopy(aligned), bad, terminal)
        bad = copy.deepcopy(evidence)
        bad["token_ids"][0] = 0
        record = copy.deepcopy(aligned)
        stream = record["token_export"]["streams"]["sft_target"]
        stream["token_ids"][0] = 0
        stream["tokens"][0]["token_id"] = 0
        with self.assertRaisesRegex(ValueError, "terminal/EOS"):
            align._validate_alignment(record, bad, terminal)

    def test_directive_prompt_anchor_and_visual_source_masks_reject(self):
        _, _, _, _, aligned, evidence = self.v1
        stream = aligned["token_export"]["streams"]["sft_target"]
        selections = {
            "directive": next(span for span in stream["spans"]
                              if span["span_type"] == "visual_route_tag"),
            "prompt": next(span for span in stream["spans"]
                           if span["span_type"] == "visual_prompt_body"),
            "anchor": next(span for span in stream["spans"]
                           if span["span_type"] == "visual_prompt_body"
                           and span.get("modality") == "video"),
        }
        for name, span in selections.items():
            with self.subTest(name=name):
                bad = copy.deepcopy(evidence)
                bad["mask"][span["token_start"]] = 1
                with self.assertRaisesRegex(ValueError, "mask=0"):
                    align._validate_alignment(copy.deepcopy(aligned), bad, [0])
        record = copy.deepcopy(aligned)
        bad = copy.deepcopy(evidence)
        zero = bad["mask"].index(0)
        record["token_export"]["streams"]["sft_target"]["tokens"][zero][
            "source_event_indices"].clear()
        with self.assertRaisesRegex(ValueError, "visual-slot source mapping"):
            align._validate_alignment(record, bad, [0])

    def test_behavior_shape_duplicate_nonfinite_and_fabricated_values_reject(self):
        _, _, _, _, _, evidence = self.v1
        mutations = {
            "duplicate": lambda value: ([value, value], "exposed"),
            "nonfinite-nan": lambda value: ([dict(value, logprobs=[float("nan")] + value["logprobs"][1:])], "exposed"),
            "nonfinite-inf": lambda value: ([dict(value, logprobs=[float("inf")] + value["logprobs"][1:])], "exposed"),
            "positive-logprob": lambda value: ([dict(value, logprobs=[0.1] + value["logprobs"][1:])], "exposed"),
            "fabricated-zeroes": lambda value: ([dict(value, logprobs=[0.0] * len(value["token_ids"]))], "not_exposed"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                values, coverage = mutate(copy.deepcopy(evidence))
                chunks = [{"behavior": value} for value in values]
                with self.assertRaises(ValueError):
                    align._behavior_from_chunks(chunks, coverage)

    def test_stale_duplicate_raw_cas_tokenizer_and_summary_tamper_reject_before_ml(self):
        def stale(root):
            path = root / "aligned/trajectories.jsonl"
            record = read_record(path)
            record["turn_index"] = 99
            update_jsonl(path, [record])
            refresh_file_hash(root, "aligned_trajectory")
            return "source record"

        def duplicate(root):
            for key in ("trajectory", "aligned_trajectory"):
                relative = "source/live/trajectories.jsonl" if key == "trajectory" \
                    else "aligned/trajectories.jsonl"
                path = root / relative
                record = read_record(path)
                update_jsonl(path, [record, record])
                refresh_file_hash(root, key)
            return "duplicate or invalid source identity"

        def raw_tamper(root):
            record = read_record(root / "source/live/trajectories.jsonl")
            path = root / "live/artifacts" / record["sse_replay"]["raw_ref"]
            raw = bytearray(path.read_bytes())
            raw[20] ^= 1
            path.write_bytes(raw)
            return "raw SSE"

        def cas_tamper(root):
            paths = [path for path in (root / "live/artifacts").iterdir()
                     if path.suffix != ".sse"]
            self.assertTrue(paths)
            raw = bytearray(paths[0].read_bytes())
            raw[-1] ^= 1
            paths[0].write_bytes(raw)
            return "CAS media"

        def tokenizer_tamper(root):
            path = root / "tokenizer/vocab.json"
            path.write_bytes(path.read_bytes() + b" ")
            return "tokenizer|eos_token"

        def summary_tamper(root):
            path = root / "summary.json"
            summary = json.loads(path.read_text())
            summary["cases"][0]["behavior"]["logprobs"][0] -= 1.0
            write_json(path, summary)
            refresh_file_hash(root, "summary")
            return "evidence hash binding"

        for name, mutate in (
            ("stale", stale), ("duplicate", duplicate), ("raw-SSE", raw_tamper),
            ("CAS", cas_tamper), ("tokenizer", tokenizer_tamper),
            ("fabricated-logprob", summary_tamper),
        ):
            with self.subTest(name=name):
                root, tokenizer_dir = self.copied(1)
                message = mutate(root)
                real_import = builtins.__import__
                def no_ml(import_name, *args, **kwargs):
                    if import_name in ("torch", "numpy") or import_name.startswith("transformers"):
                        raise AssertionError("ML import occurred before structural rejection")
                    return real_import(import_name, *args, **kwargs)
                with mock.patch("builtins.__import__", side_effect=no_ml):
                    with self.assertRaisesRegex(ValueError, message):
                        align.validate_bundle(root, tokenizer_dir, reencode=True)
                report = root / "should-not-exist.json"
                proc = subprocess.run([
                    sys.executable, "-I", str(ROOT / "python/umminfer_behavior_token_alignment.py"),
                    str(root), "--tokenizer-dir", str(tokenizer_dir), "--reencode",
                    "--report", str(report),
                ], capture_output=True, text=True, timeout=120, check=False)
                self.assertEqual(proc.returncode, 2, proc.stdout)
                self.assertFalse(report.exists())

    def test_cancelled_and_truncated_sampled_evidence_reject(self):
        with tempfile.TemporaryDirectory(prefix="umm-behavior-align-cancel-") as tmp:
            root = Path(tmp) / "cancel"
            bundle = make_bundle(root, 1, fixture.cancelled_events())
            with self.assertRaisesRegex(ValueError, "truncated/cancelled/non-stop"):
                align.validate_bundle(bundle[0], bundle[1], tokenizer=bundle[2])
        root, tokenizer_dir = self.copied(1)
        # The final frame is regenerated with finish_reason=length while the
        # normalized record remains completed, so this tests the explicit gate.
        with tempfile.TemporaryDirectory(prefix="umm-behavior-align-length-") as tmp:
            bundle = make_bundle(Path(tmp) / "length", 1, finish_reason="length")
            with self.assertRaisesRegex(ValueError, "truncated/cancelled/non-stop"):
                align.validate_bundle(bundle[0], bundle[1], tokenizer=bundle[2])

    def test_clean_text_and_source_span_tamper_reject(self):
        _, _, _, _, aligned, evidence = self.v1
        record = copy.deepcopy(aligned)
        record["clean_text"] += "tamper"
        with self.assertRaises(ValueError):
            align._validate_alignment(record, evidence, [0])
        stream = aligned["token_export"]["streams"]["sft_target"]
        bad_stream = copy.deepcopy(stream)
        bad_stream["spans"][0]["token_end"] = bad_stream["token_count"] + 1
        with self.assertRaisesRegex(ValueError, "span token range"):
            align._token_types(bad_stream)


@unittest.skipUnless(REAL_ARGS.real_tokenizer_dir,
                     "REAL_TASK42_STRONG: NOT_RUN (supply --real-tokenizer-dir)")
class RealTask42StrongTests(unittest.TestCase):
    def test_fresh_strong_report_and_real_tamper_matrix(self):
        tokenizer_dir = Path(REAL_ARGS.real_tokenizer_dir)
        expected = json.loads(REAL_REPORT.read_text())
        report = align.validate_bundle(REAL_FIXTURE, tokenizer_dir, reencode=True)
        self.assertEqual(report, expected)
        with tempfile.TemporaryDirectory(prefix="umm-task42-real-strong-") as tmp:
            output = Path(tmp) / "report.json"
            proc = subprocess.run([
                sys.executable, "-I", str(ROOT / "python/umminfer_behavior_token_alignment.py"),
                str(REAL_FIXTURE), "--tokenizer-dir", str(tokenizer_dir),
                "--reencode", "--report", str(output),
            ], capture_output=True, text=True, timeout=300, check=False)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(output.read_bytes(), REAL_REPORT.read_bytes())
            self.assertEqual(json.loads(proc.stdout), expected)

        for name in ("raw_sse", "cas_media", "token_export",
                     "summary_logprob", "tokenizer_file"):
            with self.subTest(tamper=name), tempfile.TemporaryDirectory(
                    prefix="umm-task42-real-tamper-") as tmp:
                work = Path(tmp)
                root = work / "fixture"
                tok = work / "tokenizer"
                shutil.copytree(REAL_FIXTURE, root)
                shutil.copytree(tokenizer_dir, tok)
                if name == "raw_sse":
                    record = read_record(root / "source/live/trajectories.jsonl")
                    path = root / "live/artifacts" / record["sse_replay"]["raw_ref"]
                    raw = bytearray(path.read_bytes())
                    raw[20] ^= 1
                    path.write_bytes(raw)
                elif name == "cas_media":
                    path = next(path for path in (root / "live/artifacts").iterdir()
                                if path.suffix == ".png")
                    raw = bytearray(path.read_bytes())
                    raw[-1] ^= 1
                    path.write_bytes(raw)
                elif name == "token_export":
                    path = root / "aligned/trajectories.jsonl"
                    records = [json.loads(line) for line in
                               path.read_text().splitlines() if line]
                    stream = records[0]["token_export"]["streams"]["sft_target"]
                    stream["token_ids"][0] += 1
                    stream["tokens"][0]["token_id"] += 1
                    update_jsonl(path, records)
                    refresh_file_hash(root, "aligned_trajectory")
                elif name == "summary_logprob":
                    path = root / "summary.json"
                    summary = json.loads(path.read_text())
                    summary["cases"][0]["behavior"]["logprobs"][0] -= 1.0
                    write_json(path, summary)
                    refresh_file_hash(root, "summary")
                else:
                    path = tok / "vocab.json"
                    path.write_bytes(path.read_bytes() + b" ")
                output = work / "report.json"
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ""
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                proc = subprocess.run([
                    sys.executable, "-I",
                    str(ROOT / "python/umminfer_behavior_token_alignment.py"),
                    str(root), "--tokenizer-dir", str(tok), "--reencode",
                    "--report", str(output),
                ], capture_output=True, text=True, timeout=300, env=env, check=False)
                self.assertEqual(proc.returncode, 2, proc.stdout)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
