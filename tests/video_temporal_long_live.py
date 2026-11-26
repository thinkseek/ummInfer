#!/usr/bin/env python3
"""Long-video controlled-live gate; extends tests/video_temporal_live.py.

No GPU/model request is made without UMMINFER_LIVE_OPT_IN=1 and two newly
recorded isolated loopback stacks (serial batch-size 1 and batch-four).
Synthetic endpoint tests live in tests/long_video_temporal_test.py and are
never real generation, latency, or perceptual-quality evidence.

Offline:
  python3 -I tests/video_temporal_long_live.py --self-check
  python3 -I tests/video_temporal_long_live.py --check-evidence evidence-dir

Live arguments are documented in docs/video-temporal-long.md. Only physical
GPUs 0, 1, and 7 can pass provenance validation. Budget 8 is the sole public
native live budget; CPU oracle coverage also exercises 4 and 16.
"""

from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import umminfer_long_video_fixture as fixture  # noqa: E402
import umminfer_long_video_live as long_live  # noqa: E402
import video_temporal_live as short  # noqa: E402


def budgets(text):
    try:
        values = [int(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("budgets must be comma-separated integers") from error
    return values


def self_check():
    long_live.validate_budgets([8])
    for unsupported in ([4], [16], [4, 8, 16], [8, 8]):
        try:
            long_live.validate_budgets(unsupported)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsupported live budget accepted: {unsupported}")
    for budget in fixture.CPU_BUDGETS:
        selected = fixture.select_temporal([(0.0, 0.0)] * 57, budget)
        assert len(selected) == budget and selected[0] == 0 and selected[-1] == 56
    assert long_live.answer_hit("blanco y negro", fixture.CASES[3]) is True
    assert long_live.answer_hit("blanco solamente", fixture.CASES[3]) is False
    print("SELF-CHECK PASS: long budgets, anchors, multilingual expectations, live opt-in boundary")
    print("REAL_GENERATION: NOT_RUN")
    print("LIVE_LATENCY: NOT_RUN")
    print("PERCEPTUAL_QUALITY: NOT_RUN")
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--self-check", action="store_true")
    result.add_argument("--check-evidence", default="")
    result.add_argument("--manifest", default=str(ROOT / "benchmarks" /
                                                   "video-temporal-long" / "manifest.json"))
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
            report = long_live.validate_bundle(args.check_evidence)
        except (OSError, ValueError) as error:
            print(f"EVIDENCE FAIL: {error}", file=sys.stderr)
            return 1
        print({"result": "PASS", "evidence_class": report["evidence_class"],
               "gate_pass": report["gate_pass"], "claims": report["claims"]})
        return 0 if report["gate_pass"] else 1
    if os.environ.get("UMMINFER_LIVE_OPT_IN") != "1":
        print("REAL_GENERATION: NOT_RUN")
        print("LIVE_LATENCY: NOT_RUN")
        print("PERCEPTUAL_QUALITY: NOT_RUN")
        print("Set UMMINFER_LIVE_OPT_IN=1 and supply two newly recorded isolated stacks.")
        return 0
    required = ("serial_url", "batch_url", "serial_native_log", "batch_native_log",
                "serial_provenance", "batch_provenance", "run_id", "out_dir")
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser().error(f"live mode missing arguments: {missing}")
    try:
        return long_live.execute_live(args, short)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"LIVE FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
