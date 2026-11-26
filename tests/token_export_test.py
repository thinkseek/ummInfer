#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic CPU-only tests for optional trajectory-v2 token/span export.

Default tests use a deterministic fixture tokenizer and import no ML framework.
For the optional local fast-tokenizer gate, supply all of:
  --real-tokenizer-dir DIR --real-tokenizer-id ID
  --real-tokenizer-revision REV --real-tokenization-version VERSION
No model inference, GPU, network, or serving process is used.
"""

import argparse
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import narrative_rl_fixtures as fixture  # noqa: E402
import umminfer_joint_sft_live as live_sft  # noqa: E402
import umminfer_joint_trainer as stage_c  # noqa: E402
import umminfer_rl_trajectory as traj  # noqa: E402
import umminfer_rl_worker as worker  # noqa: E402
import umminfer_sft_dataset as dataset  # noqa: E402
import umminfer_sse_replay as replay  # noqa: E402
import umminfer_token_export as tokens  # noqa: E402


class FixtureTokenizer:
    """Longest-match Unicode tokenizer with exact, no-cleanup token decode."""

    def __init__(self, special_pieces=()):
        self.special_pieces = tuple(sorted(
            set(special_pieces), key=lambda value: (-len(value), value)))
        self.special_ids = {
            piece: 0x110001 + index
            for index, piece in enumerate(self.special_pieces)
        }
        self.id_pieces = {value: key for key, value in self.special_ids.items()}
        self.unk_token_id = 0
        self.calls = []

    def __len__(self):
        return 0x110001 + len(self.special_pieces)

    def __call__(self, text, **kwargs):
        self.calls.append((text, copy.deepcopy(kwargs)))
        ids = []
        offsets = []
        cursor = 0
        while cursor < len(text):
            piece = next((candidate for candidate in self.special_pieces
                          if text.startswith(candidate, cursor)), None)
            if piece is None:
                piece = text[cursor]
                token_id = ord(piece) + 1
            else:
                token_id = self.special_ids[piece]
            ids.append(token_id)
            offsets.append((cursor, cursor + len(piece)))
            cursor += len(piece)
        return {"input_ids": ids, "offset_mapping": offsets}

    def decode(self, ids, **kwargs):
        self.assert_decode_kwargs = copy.deepcopy(kwargs)
        pieces = []
        for token_id in ids:
            if token_id in self.id_pieces:
                pieces.append(self.id_pieces[token_id])
            elif 1 <= token_id <= 0x110000:
                pieces.append(chr(token_id - 1))
            else:
                pieces.append("<?>")
        return "".join(pieces)


def fixture_tokenizer():
    return FixtureTokenizer((
        "[[image: ", "[[video: ", " | source=slot-0]]", "]]",
        "<spoken>", "</spoken>", ", a", "灯笼 🏮",
    ))


def _record(root, events):
    root = Path(root)
    recorder = traj.TrajectoryRecorder(root / "trajectory.jsonl")
    raw = fixture.sse_bytes(events)
    capture = traj.capture_sse_response(
        io.BytesIO(raw), clock=replay.LogicalTickClock())
    details = {}
    request_id, events, output, state, error = traj.parse_sse_stream(
        io.BytesIO(capture.raw_bytes), artifact_dir=recorder.artifact_dir,
        details=details, capture=capture)
    record = recorder.record(
        session_id="token-fixture", request_id=request_id,
        endpoint="/v1/chat/completions",
        inputs=[{"kind": "text", "ref": None, "media_type": None,
                 "text": "tell a multilingual visual story",
                 "sha256": traj.sha256_text("tell a multilingual visual story")}],
        params={"model": "fixture", "stream": True}, events=events,
        terminal_state=state, error=error, output_text=output,
        artifacts=details["artifacts"], capture=capture)
    return recorder, record


def _tokenizer_dir(root):
    root = Path(root) / "tokenizer"
    root.mkdir(parents=True, exist_ok=True)
    (root / "tokenizer.json").write_text(
        '{"format":"deterministic-unicode-fixture","version":1}\n',
        encoding="utf-8")
    (root / "config.json").write_text(
        '{"add_special_tokens":false,"padding":false,"truncation":false}\n',
        encoding="utf-8")
    return root


def _attach(record, root, tokenizer=None, expected_target=True):
    tokenizer = tokenizer or fixture_tokenizer()
    target = None
    if expected_target:
        target = dataset.reconstruct_director_target(record)[0]
    return tokens.attach_token_export(
        record, tokenizer, tokenizer_dir=_tokenizer_dir(root),
        tokenizer_id="fixture-unicode-bpe",
        tokenizer_revision="fixture-revision-1",
        tokenization_version="fixture-tokenization-v1",
        sft_target=target)


def _multilingual_record(root):
    first = "<spoken>黎明🌅 é"
    second = " 字。</spoken> 尾🧪"
    clean = first + second
    events = [
        fixture.envelope(0, "vlm_delta", first),
        fixture.visual_slot(
            1, 0, "image", "灯笼 🏮 é", len(first.encode("utf-8"))),
        fixture.envelope(2, "vlm_delta", second),
        fixture.envelope(3, "complete", clean, legs=[{
            "stage": "image", "slot_id": "slot-0", "slot_index": 0,
            "state": "completed",
        }]),
    ]
    return _record(root, events)


def _assert_mask_equivalence(testcase, record, exported):
    target = dataset.reconstruct_director_target(record)[0]
    mask = stage_c.build_span_loss_mask(target)
    stream = exported["token_export"]["streams"]["sft_target"]
    ids = stream["token_ids"]
    offsets = [(token["char_start"], token["char_end"])
               for token in stream["tokens"]]
    labels, weights, annotations = live_sft._token_loss_mask(
        offsets, target_start=0, target=target, token_ids=ids, mask_info=mask)
    testcase.assertEqual(labels, ids)
    testcase.assertEqual(weights, stream["sft_mask"]["token_weights"])
    testcase.assertEqual(
        stream["sft_mask"]["annotations_sha256"],
        stage_c.sha256_bytes(stage_c.canonical_json(annotations)))
    testcase.assertTrue(any(item["includes_terminal_delimiter"]
                            for item in annotations))
    for index, span in enumerate(mask["spans"]):
        exported_span = stream["spans"][index]
        testcase.assertEqual(exported_span["span_type"], span["span_type"])
        testcase.assertEqual(exported_span["modality"], span["modality"])
        testcase.assertEqual(exported_span["char_start"], span["start"])
        testcase.assertEqual(exported_span["char_end"], span["end"])
        testcase.assertEqual(
            target.encode("utf-8")[exported_span["start"]:exported_span["end"]]
            .decode("utf-8"), span["text"])


def _write_attached(root, record):
    path = Path(root) / "attached.jsonl"
    path.write_bytes(tokens._canonical(record) + b"\n")
    return path


def _refresh_export_hash(record):
    block = record["token_export"]
    unsigned = {key: value for key, value in block.items() if key != "export_sha256"}
    block["export_sha256"] = tokens._sha_bytes(tokens._canonical(unsigned))


class TokenExportTests(unittest.TestCase):
    def test_deterministic_explicit_anchor_source_frames_and_worker_sft_validation(self):
        with tempfile.TemporaryDirectory(prefix="umm-token-linked-") as tmp:
            root = Path(tmp)
            recorder_a, record_a = _record(root / "a", fixture.linked_events())
            recorder_b, record_b = _record(root / "b", fixture.linked_events())
            untouched = copy.deepcopy(record_a)
            exported_a = _attach(record_a, root / "export-a")
            exported_b = _attach(record_b, root / "export-b")
            self.assertEqual(record_a, untouched)
            self.assertEqual(exported_a, exported_b)
            self.assertEqual(tokens._canonical(exported_a), tokens._canonical(exported_b))
            self.assertNotIn(str(root), json.dumps(exported_a))
            self.assertNotIn('"data":', json.dumps(exported_a))

            block = exported_a["token_export"]
            self.assertEqual(block["format"], tokens.TOKEN_EXPORT_FORMAT)
            self.assertEqual(block["policy"]["padding"], "none")
            self.assertEqual(block["policy"]["truncation"], "none")
            self.assertEqual(block["policy"]["unknown_tokens"], "reject")
            self.assertEqual(len(block["tokenizer"]["available_files"]), 2)
            for stream in block["streams"].values():
                text = (record_a["clean_text"] if stream["source"] == "clean_text"
                        else dataset.reconstruct_director_target(record_a)[0])
                raw = text.encode("utf-8")
                self.assertEqual(stream["text_sha256"], tokens._sha_bytes(raw))
                cursor = 0
                decoded = []
                for token_id, token in zip(stream["token_ids"], stream["tokens"]):
                    self.assertEqual(token["token_id"], token_id)
                    self.assertEqual(token["start"], cursor)
                    self.assertGreater(token["end"], token["start"])
                    piece = raw[token["start"]:token["end"]]
                    decoded.append(piece)
                    self.assertEqual(token["piece_sha256"], tokens._sha_bytes(piece))
                    for event_index, frame_index in zip(
                            token["source_event_indices"], token["source_frame_indices"]):
                        self.assertEqual(
                            record_a["events"][event_index]["source_event_index"],
                            frame_index)
                    cursor = token["end"]
                self.assertEqual(cursor, len(raw))
                self.assertEqual(b"".join(decoded), raw)

            target_stream = block["streams"]["sft_target"]
            target = dataset.reconstruct_director_target(record_a)[0]
            self.assertIn("[[video: the lantern sways in the wind | source=slot-0]]",
                          target)
            route_spans = [span for span in target_stream["spans"]
                           if span["span_type"] == "visual_route_tag"]
            body_spans = [span for span in target_stream["spans"]
                          if span["span_type"] == "visual_prompt_body"]
            self.assertGreaterEqual(len(route_spans), 4)
            self.assertTrue(body_spans)
            self.assertGreaterEqual(
                target_stream["sft_mask"]["terminal_delimiter_tokens"], 2)
            # The fixture token ", a" crosses two clean deltas and must retain both.
            self.assertTrue(any(
                len(token["source_event_indices"]) == 2
                for token in block["streams"]["clean_text"]["tokens"]))

            path = _write_attached(root / "a", exported_a)
            loaded = worker.validate_trajectory(path)
            self.assertEqual(loaded, [exported_a])
            ok, category, message, stats = dataset.verify_record_integrity(
                exported_a, artifact_dir=recorder_a.artifact_dir,
                require_artifact_dir=True)
            self.assertTrue(ok, (category, message))
            self.assertEqual(stats["checked_token_exports"], 1)
            self.assertEqual(stats["checked_replay_refs"], 1)
            result = tokens.validate_record_token_export(
                exported_a, expected_sft_target=target)
            self.assertEqual(result["status"], "PASS")
            self.assertFalse(result["tokenizer_reencoded"])
            strong = tokens.validate_record_token_export(
                exported_a, tokenizer=fixture_tokenizer(),
                tokenizer_dir=_tokenizer_dir(root / "export-a"),
                expected_tokenizer_id="fixture-unicode-bpe",
                expected_tokenizer_revision="fixture-revision-1",
                expected_tokenization_version="fixture-tokenization-v1",
                expected_sft_target=target)
            self.assertEqual(strong["status"], "PASS")
            self.assertTrue(strong["tokenizer_reencoded"])
            _assert_mask_equivalence(self, record_a, exported_a)

    def test_multilingual_emoji_combining_spoken_and_exact_byte_offsets(self):
        with tempfile.TemporaryDirectory(prefix="umm-token-utf8-") as tmp:
            recorder, record = _multilingual_record(tmp)
            exported = _attach(record, tmp)
            stream = exported["token_export"]["streams"]["sft_target"]
            target = dataset.reconstruct_director_target(record)[0]
            raw = target.encode("utf-8")
            self.assertIn("🌅", record["clean_text"])
            self.assertIn("é", record["clean_text"])
            self.assertTrue(any(span["span_type"] == "spoken_style"
                                for span in stream["spans"]))
            self.assertTrue(any(span["span_type"] == "visual_prompt_body"
                                for span in stream["spans"]))
            for token in stream["tokens"]:
                raw[token["start"]:token["end"]].decode("utf-8")
            # A multibyte prompt token still maps to the visual-slot SSE frame.
            prompt_span = next(span for span in stream["spans"]
                               if span["span_type"] == "visual_prompt_body")
            for token in stream["tokens"][prompt_span["token_start"]:
                                          prompt_span["token_end"]]:
                self.assertIn(1, token["source_event_indices"])
                self.assertIn(1, token["source_frame_indices"])
            _assert_mask_equivalence(self, record, exported)
            path = _write_attached(tmp, exported)
            worker.validate_trajectory(path)
            ok, category, message, _ = dataset.verify_record_integrity(
                exported, artifact_dir=recorder.artifact_dir,
                require_artifact_dir=True)
            self.assertTrue(ok, (category, message))

    def test_cancellation_and_legacy_absence_are_preserved(self):
        with tempfile.TemporaryDirectory(prefix="umm-token-cancel-") as tmp:
            recorder, cancelled = _record(Path(tmp) / "cancel", fixture.cancelled_events())
            exported = _attach(cancelled, Path(tmp) / "export")
            self.assertEqual(exported["terminal_state"], "cancelled")
            result = tokens.validate_record_token_export(exported)
            self.assertEqual(result["status"], "PASS")
            path = _write_attached(Path(tmp) / "cancel", exported)
            worker.validate_trajectory(path)

            legacy = {
                "schema_version": 1, "session_id": "legacy", "request_id": "old",
                "turn_index": 0, "endpoint": "/v1/chat/completions",
                "inputs": [], "request_params": {},
                "events": [{"seq": 0, "type": "output_delta", "text": "old"}],
                "terminal_state": "completed", "error": None,
                "output_text": "old", "output_sha256": traj.sha256_text("old"),
            }
            self.assertEqual(tokens.validate_record_token_export(legacy)["status"],
                             "ABSENT")
            self.assertEqual(tokens.validate_record_token_export(cancelled)["status"],
                             "ABSENT")
            self.assertNotIn("token_export", cancelled)
            legacy_path = Path(tmp) / "legacy.jsonl"
            legacy_path.write_bytes(tokens._canonical(legacy) + b"\n")
            self.assertEqual(worker.validate_trajectory(legacy_path), [legacy])
            legacy_with_export = copy.deepcopy(legacy)
            legacy_with_export["token_export"] = {}
            with self.assertRaises(tokens.TokenExportError):
                tokens.validate_record_token_export(legacy_with_export)
            legacy_path.write_bytes(tokens._canonical(legacy_with_export) + b"\n")
            with self.assertRaises(ValueError):
                worker.validate_trajectory(legacy_path)
            ok, category, _, _ = dataset.verify_record_integrity(legacy_with_export)
            self.assertFalse(ok)
            self.assertEqual(category, "token_export_tampered")

            cli = subprocess.run([
                sys.executable, "-I", str(ROOT / "python" / "umminfer_token_export.py"),
                str(path), "--validate-only",
            ], capture_output=True, text=True, timeout=60)
            self.assertEqual(cli.returncode, 0, cli.stdout + cli.stderr)
            payload = json.loads(cli.stdout)
            self.assertEqual(payload["validated"], 1)
            self.assertEqual(payload["tokenizer_reencoded"], 0)

    def test_malformed_offsets_spans_ids_sources_hashes_and_config_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix="umm-token-malformed-") as tmp:
            recorder, record = _record(Path(tmp) / "base", fixture.linked_events())
            valid = _attach(record, tmp)
            path = _write_attached(Path(tmp) / "base", valid)

            mutations = []

            def mutation(name, function):
                def apply(candidate):
                    function(candidate)
                    return candidate
                mutations.append((name, apply))

            def stream(candidate, kind="sft_target"):
                return candidate["token_export"]["streams"][kind]

            def set_first_id(candidate, value):
                stream(candidate)["token_ids"][0] = value
                stream(candidate)["tokens"][0]["token_id"] = value

            mutation("negative-id", lambda c: set_first_id(c, -1))
            mutation("boolean-id", lambda c: set_first_id(c, True))
            mutation("float-id", lambda c: set_first_id(
                c, float(stream(c)["token_ids"][0])))
            mutation("out-of-vocab", lambda c: set_first_id(c, 10**12))
            mutation("unknown-id", lambda c: set_first_id(c, 0))
            mutation("boolean-token-index", lambda c: stream(c)["tokens"][0]
                     .__setitem__("index", False))
            mutation("float-char-offset", lambda c: stream(c)["tokens"][0]
                     .__setitem__("char_start", float(stream(c)["tokens"][0]["char_start"])))
            mutation("boolean-source-frame", lambda c: stream(c)["tokens"][0]
                     ["source_frame_indices"].__setitem__(0, False))
            mutation("boolean-source-event", lambda c: stream(c)["tokens"][0]
                     ["source_event_indices"].__setitem__(0, False))
            mutation("boolean-format-version", lambda c: c["token_export"]
                     .__setitem__("format_version", True))
            mutation("offset-gap", lambda c: stream(c)["tokens"][0].__setitem__("start", 1))
            mutation("offset-overlap", lambda c: stream(c)["tokens"][1].__setitem__(
                "start", stream(c)["tokens"][0]["start"]))
            mutation("offset-range", lambda c: stream(c)["tokens"][-1].__setitem__(
                "end", stream(c)["text_bytes"] + 1))
            mutation("span-gap", lambda c: stream(c)["spans"][1].__setitem__("start", 0))
            mutation("span-range", lambda c: stream(c)["spans"][-1].__setitem__(
                "end", stream(c)["text_bytes"] + 1))
            mutation("span-token-range", lambda c: stream(c)["spans"][0].__setitem__(
                "token_end", 10**6))
            mutation("piece-hash", lambda c: stream(c)["tokens"][0].__setitem__(
                "piece_sha256", "0" * 64))
            mutation("token-list-hash", lambda c: stream(c).__setitem__(
                "tokens_sha256", "0" * 64))
            mutation("mask-weight", lambda c: stream(c)["sft_mask"]
                     ["token_weights"].__setitem__(0, 0.0))
            mutation("source-frame", lambda c: stream(c)["tokens"][0]
                     ["source_frame_indices"].clear())
            mutation("tokenizer-id", lambda c: c["token_export"]["tokenizer"]
                     .__setitem__("id", "changed"))
            mutation("tokenizer-revision", lambda c: c["token_export"]["tokenizer"]
                     .__setitem__("revision", "changed"))
            mutation("tokenization-version", lambda c: c["token_export"]["tokenizer"]
                     .__setitem__("tokenization_version", "changed"))
            mutation("padding-policy", lambda c: c["token_export"]["policy"]
                     .__setitem__("padding", "longest"))
            mutation("truncation-policy", lambda c: c["token_export"]["policy"]
                     .__setitem__("truncation", "left"))
            mutation("unknown-policy", lambda c: c["token_export"]["policy"]
                     .__setitem__("unknown_tokens", "allow"))
            mutation("boolean-policy-impersonation", lambda c: c["token_export"]["policy"]
                     .__setitem__("add_special_tokens", 0))
            mutation("available-file-hash", lambda c: c["token_export"]["tokenizer"]
                     ["available_files"][0].__setitem__("sha256", "0" * 64))
            mutation("source-record-hash", lambda c: c["token_export"].__setitem__(
                "source_record_sha256", "0" * 64))
            mutation("export-hash", lambda c: c["token_export"].__setitem__(
                "export_sha256", "0" * 64))
            mutation("source-text-change", lambda c: c.__setitem__(
                "clean_text", c["clean_text"] + "tamper"))

            for name, apply in mutations:
                with self.subTest(name=name):
                    bad = apply(copy.deepcopy(valid))
                    # Re-sign the outer block so structural checks, not only the
                    # final export digest, must reject each malformed mutation.
                    if name != "export-hash":
                        _refresh_export_hash(bad)
                    with self.assertRaises(tokens.TokenExportError):
                        tokens.validate_record_token_export(bad)
                    path.write_bytes(tokens._canonical(bad) + b"\n")
                    with self.assertRaises(ValueError):
                        worker.validate_trajectory(path)
                    ok, category, _, _ = dataset.verify_record_integrity(
                        bad, artifact_dir=recorder.artifact_dir,
                        require_artifact_dir=True)
                    self.assertFalse(ok)
                    if name == "source-text-change":
                        # The outer trajectory digest rejects this before token validation.
                        self.assertEqual(category, "integrity_error")
                    else:
                        self.assertEqual(category, "token_export_tampered")

            path.write_bytes(tokens._canonical(valid) + b"\n")
            worker.validate_trajectory(path)
            with self.assertRaisesRegex(tokens.TokenExportError, "tokenizer id changed"):
                tokens.validate_record_token_export(
                    valid, expected_tokenizer_id="different")
            with self.assertRaisesRegex(tokens.TokenExportError, "tokenizer revision changed"):
                tokens.validate_record_token_export(
                    valid, expected_tokenizer_revision="different")
            with self.assertRaisesRegex(tokens.TokenExportError, "tokenization version changed"):
                tokens.validate_record_token_export(
                    valid, expected_tokenization_version="different")

            tokenizer_file = Path(tmp) / "tokenizer" / "tokenizer.json"
            original = tokenizer_file.read_bytes()
            try:
                tokenizer_file.write_bytes(original + b"tamper\n")
                with self.assertRaisesRegex(
                        tokens.TokenExportError, "available-file hashes or bytes changed"):
                    tokens.validate_record_token_export(
                        valid, tokenizer_dir=Path(tmp) / "tokenizer")
            finally:
                tokenizer_file.write_bytes(original)

    def test_export_rejects_unknown_tokens_bad_decode_and_target_disagreement(self):
        with tempfile.TemporaryDirectory(prefix="umm-token-tokenizer-errors-") as tmp:
            _, record = _record(Path(tmp) / "base", fixture.linked_events())

            class UnknownTokenizer(FixtureTokenizer):
                def __call__(self, text, **kwargs):
                    encoded = super().__call__(text, **kwargs)
                    if encoded["input_ids"]:
                        encoded["input_ids"][0] = self.unk_token_id
                    return encoded

            with self.assertRaisesRegex(tokens.TokenExportError, "unknown token"):
                _attach(record, Path(tmp) / "unknown", UnknownTokenizer())

            class BadDecodeTokenizer(FixtureTokenizer):
                def decode(self, ids, **kwargs):
                    return "not-the-source"

            with self.assertRaisesRegex(tokens.TokenExportError, "decode differs"):
                _attach(record, Path(tmp) / "decode", BadDecodeTokenizer())

            with self.assertRaisesRegex(tokens.TokenExportError, "SFT dataset target"):
                tokens.attach_token_export(
                    record, fixture_tokenizer(), tokenizer_dir=_tokenizer_dir(
                        Path(tmp) / "target"),
                    tokenizer_id="fixture-unicode-bpe",
                    tokenizer_revision="fixture-revision-1",
                    tokenization_version="fixture-tokenization-v1",
                    sft_target="noncanonical target")


def _parse_real_gate_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--real-tokenizer-dir")
    parser.add_argument("--real-tokenizer-id")
    parser.add_argument("--real-tokenizer-revision")
    parser.add_argument("--real-tokenization-version")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    values = (args.real_tokenizer_dir, args.real_tokenizer_id,
              args.real_tokenizer_revision, args.real_tokenization_version)
    if any(value is not None for value in values) and not all(
            value is not None for value in values):
        parser.error("the real-tokenizer gate requires all four options")
    return args


REAL_ARGS = _parse_real_gate_args()


class RealTokenizerFixtureTests(unittest.TestCase):
    @unittest.skipUnless(REAL_ARGS.real_tokenizer_dir,
                         "REAL_TOKENIZER_FIXTURE: NOT_RUN (no local fixture supplied)")
    def test_real_fast_tokenizer_reencode_decode_and_file_binding(self):
        tokenizer = tokens._load_real_tokenizer(REAL_ARGS.real_tokenizer_dir)
        with tempfile.TemporaryDirectory(prefix="umm-token-real-") as tmp:
            recorder, record = _record(Path(tmp) / "base", fixture.linked_events())
            target = dataset.reconstruct_director_target(record)[0]
            exported = tokens.attach_token_export(
                record, tokenizer,
                tokenizer_dir=REAL_ARGS.real_tokenizer_dir,
                tokenizer_id=REAL_ARGS.real_tokenizer_id,
                tokenizer_revision=REAL_ARGS.real_tokenizer_revision,
                tokenization_version=REAL_ARGS.real_tokenization_version,
                sft_target=target)
            result = tokens.validate_record_token_export(
                exported, tokenizer=tokenizer,
                tokenizer_dir=REAL_ARGS.real_tokenizer_dir,
                expected_tokenizer_id=REAL_ARGS.real_tokenizer_id,
                expected_tokenizer_revision=REAL_ARGS.real_tokenizer_revision,
                expected_tokenization_version=REAL_ARGS.real_tokenization_version,
                expected_sft_target=target)
            self.assertTrue(result["tokenizer_reencoded"])
            self.assertEqual(result["status"], "PASS")
            _assert_mask_equivalence(self, record, exported)
            ok, category, message, _ = dataset.verify_record_integrity(
                exported, artifact_dir=recorder.artifact_dir,
                require_artifact_dir=True)
            self.assertTrue(ok, (category, message))


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False).result
    if REAL_ARGS.real_tokenizer_dir:
        print("REAL_TOKENIZER_FIXTURE: " + ("PASS" if result.wasSuccessful() else "FAIL"))
    else:
        print("REAL_TOKENIZER_FIXTURE: NOT_RUN (supply all four --real-tokenizer-* options)")
    raise SystemExit(0 if result.wasSuccessful() else 1)
