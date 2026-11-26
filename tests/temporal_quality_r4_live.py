#!/usr/bin/env python3
"""Task #47 R4 audit and controlled serial GPU-0 intervention harness.

Offline preparation:
  python3 -I tests/temporal_quality_r4_live.py --prepare \
    --bundle benchmarks/video-temporal-long \
    --r4-evidence /path/to/sealed-r4-evidence --out-dir /tmp/task47-plan

After running the host-only qwen38_prompt_position_audit tool:
  python3 -I tests/temporal_quality_r4_live.py --finalize-positions \
    --plan /tmp/task47-plan --position-ids /tmp/task47-position-ids.json

Live execution requires UMMINFER_LIVE_OPT_IN=1, a fresh serial GPU-0 provenance
file, and an isolated loopback URL. See docs/video-temporal-long.md. No default
invocation contacts a GPU or service. --check-evidence independently reopens a
sealed result in a fresh process; --check-evidence-shallow skips media decoding.
"""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import umminfer_temporal_quality as quality  # noqa: E402
import video_temporal_live as short  # noqa: E402


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--prepare", action="store_true")
    result.add_argument("--finalize-positions", action="store_true")
    result.add_argument("--verify-plan", action="store_true")
    result.add_argument("--self-check", action="store_true")
    result.add_argument("--bundle", default=str(ROOT / "benchmarks/video-temporal-long"))
    result.add_argument("--r4-evidence", default="")
    result.add_argument("--position-ids", default="")
    result.add_argument("--plan", default="")
    result.add_argument("--check-evidence", default="")
    result.add_argument("--check-evidence-shallow", action="store_true")
    result.add_argument("--reseal-evidence", action="store_true")
    result.add_argument("--url", default="")
    result.add_argument("--native-log", default="")
    result.add_argument("--provenance", default="")
    result.add_argument("--source-root", default=str(ROOT))
    result.add_argument("--run-id", default="")
    result.add_argument("--out-dir", default="")
    return result


def self_check():
    case = quality.fixture.CASES[0]
    duration = 224.5
    nominal = quality.uniform_nominal_timestamps(duration)
    assert nominal == [0.0, 28.0625, 56.125, 84.1875,
                       112.25, 140.3125, 168.375, 196.4375]
    assert quality.pair_first(nominal) == [0.0, 56.125, 112.25, 168.375]
    assert quality.timestamps_for_arm(case, "temporal_ordinal_pts", duration) == \
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    print("SELF-CHECK PASS: fixed arms, nominal/source/ordinal PTS, opt-in CLI boundary")
    print("REAL_GENERATION: NOT_RUN")
    print("LIVE_LATENCY: NOT_RUN")
    print("PERCEPTUAL_QUALITY: NOT_RUN")
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.self_check:
            return self_check()
        if args.prepare:
            if not args.out_dir or not args.r4_evidence:
                parser().error("--prepare requires --r4-evidence and --out-dir")
            plan = quality.prepare_plan(args.bundle, args.out_dir,
                                        r4_evidence=args.r4_evidence)
            print({"result": "PASS", "cases": len(plan["cases"]), "arms": plan["arms"],
                   "out_dir": str(Path(args.out_dir).resolve()),
                   "real_generation": "NOT_RUN", "perceptual_quality": "NOT_RUN"})
            return 0
        if args.finalize_positions:
            if not args.plan or not args.position_ids:
                parser().error("--finalize-positions requires --plan and --position-ids")
            entries = quality.finalize_positions(args.plan, args.position_ids)
            print({"result": "PASS", "independent_position_entries": len(entries),
                   "real_generation": "NOT_RUN"})
            return 0
        if args.verify_plan:
            if not args.plan:
                parser().error("--verify-plan requires --plan")
            plan, oracle = quality.validate_plan(args.plan, deep=True, require_positions=True)
            print({"result": "PASS", "cases": len(plan["cases"]),
                   "independent_position_entries": len(oracle), "real_generation": "NOT_RUN"})
            return 0
        if args.reseal_evidence:
            target = args.check_evidence or args.plan
            if not target:
                parser().error("--reseal-evidence requires --check-evidence DIR")
            report = quality.reseal_evidence(target)
            print({"result": "PASS", "resealed": True, "gate_pass": report["gate_pass"],
                   "records": len(report["records"]), "claims": report["claims"]})
            return 0 if report["gate_pass"] else 1
        if args.check_evidence or args.check_evidence_shallow:
            target = args.check_evidence or args.plan
            if not target:
                parser().error("evidence verification requires --check-evidence DIR")
            report = quality.validate_bundle(target, deep=not args.check_evidence_shallow)
            print({"result": "PASS", "gate_pass": report["gate_pass"],
                   "evidence_class": report["evidence_class"], "claims": report["claims"],
                   "answer_outcomes": report["answer_outcomes"]})
            return 0 if report["gate_pass"] else 1
        required = ("plan", "url", "native_log", "provenance", "run_id", "out_dir")
        missing = [name for name in required if not getattr(args, name)]
        if missing:
            print("REAL_GENERATION: NOT_RUN")
            print("LIVE_LATENCY: NOT_RUN")
            print("PERCEPTUAL_QUALITY: NOT_RUN")
            parser().error(f"live mode missing arguments: {missing}")
        return quality.execute_live(args, short)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"TASK47 FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
