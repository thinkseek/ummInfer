#!/usr/bin/env python3
"""Opt-in temporal-clean-r1 long-video fixture and controlled-live gate.

Offline:
  python3 -I tests/temporal_clean_live.py --self-check
  python3 -I tests/temporal_clean_live.py --check-evidence /path/to/evidence

Live execution requires UMMINFER_LIVE_OPT_IN=1, a committed frozen fixture, and
new isolated serial GPU 0 / batch-four GPU 1 loopback stacks. GPU 7, GPUs 2-6,
and resident port 43301 are forbidden. See docs/video-temporal-clean.md.
"""

from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import umminfer_long_video_live as long_live  # noqa: E402
import umminfer_temporal_clean_fixture as fixture  # noqa: E402
import umminfer_temporal_clean_live as clean  # noqa: E402
import video_temporal_live as short  # noqa: E402


def budgets(text):
    try:
        return [int(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("budgets must be comma-separated integers") from error


def self_check():
    for supported in ([4], [8], [16], [4, 8, 16]):
        clean.validate_budgets(supported)
    for unsupported in ([8, 8], [0], [3], [17], [4.0], [True], []):
        try:
            clean.validate_budgets(unsupported)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsupported clean live budget accepted: {unsupported}")
    body = clean.build_body_after_context(
        "data:video/mp4;base64,AA==", "What shape?", "temporal", "qwen3-native",
        json_request_contract(), 16)
    assert [part["type"] for part in body["messages"][0]["content"]] == ["video_url", "text"]
    assert body["seed"] == 510051 and body["max_tokens"] == 64
    assert body["messages"][0]["content"][0]["video_url"]["frame_selection"]["max_frames"] == 16
    default = clean.build_body_after_context(
        "data:video/mp4;base64,AA==", "What shape?", "default", "qwen3-native",
        json_request_contract())
    assert "frame_selection" not in default["messages"][0]["content"][0]["video_url"]
    print("SELF-CHECK PASS: explicit 4/8/16, unchanged default, question-after-context, seed")
    print("REAL_GENERATION: NOT_RUN")
    print("LIVE_LATENCY: NOT_RUN")
    print("FIXTURE_SUBSTRING_ACCURACY: NOT_RUN")
    print("PERCEPTUAL_QUALITY: NOT_RUN")
    return 0


def json_request_contract():
    return {"temperature": 0.0, "top_p": 1.0, "seed": 510051, "max_tokens": 64}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--self-check", action="store_true")
    result.add_argument("--check-evidence", default="")
    result.add_argument("--manifest", default=str(fixture.DEFAULT_BUNDLE / "manifest.json"))
    result.add_argument("--freeze", default=str(fixture.DEFAULT_FREEZE))
    result.add_argument("--budgets", type=budgets, default=[8])
    result.add_argument("--serial-url", default="")
    result.add_argument("--batch-url", default="")
    result.add_argument("--serial-native-log", default="")
    result.add_argument("--batch-native-log", default="")
    result.add_argument("--serial-provenance", default="")
    result.add_argument("--batch-provenance", default="")
    result.add_argument("--source-root", default=str(ROOT))
    result.add_argument("--run-id", default="")
    result.add_argument("--out-dir", default="")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.self_check:
        return self_check()
    if args.check_evidence:
        try:
            report = clean.validate_bundle(args.check_evidence, short, deep=True)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            print(f"CLEAN EVIDENCE FAIL: {error}", file=sys.stderr)
            return 1
        print({"result": "PASS", "evidence_class": report["evidence_class"],
               "gate_pass": report["gate_pass"], "claims": report["claims"],
               "quality_answer_outcomes": report["answer_outcomes"].get("quality_scored", {})})
        return 0 if report["gate_pass"] else 1
    if os.environ.get("UMMINFER_LIVE_OPT_IN") != "1":
        print("REAL_GENERATION: NOT_RUN")
        print("LIVE_LATENCY: NOT_RUN")
        print("FIXTURE_SUBSTRING_ACCURACY: NOT_RUN")
        print("PERCEPTUAL_QUALITY: NOT_RUN")
        print("Set UMMINFER_LIVE_OPT_IN=1 and supply two newly recorded isolated GPU 0/1 stacks.")
        return 0
    required = ("serial_url", "batch_url", "serial_native_log", "batch_native_log",
                "serial_provenance", "batch_provenance", "run_id", "out_dir")
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser().error(f"live mode missing arguments: {missing}")
    try:
        return clean.execute_live(args, short)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"CLEAN LIVE FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
