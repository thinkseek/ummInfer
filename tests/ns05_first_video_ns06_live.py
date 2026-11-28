#!/usr/bin/env python3
"""Run the frozen NS-06 After arm with a new, explicitly bound execution commit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import ns06_cross_slot_live as frozen  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--preregistration", type=Path,
                        default=ROOT / "benchmarks/cross-slot-visual-continuity/ns06-preregistration.json")
    for name in ("server", "vlm", "asr", "tts", "wan", "flux"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("vlm-model", "asr-model", "flux-model", "tts-model", "wan-model",
                 "sketch", "voice"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--web-root", type=Path, default=ROOT / "web")
    parser.add_argument("--verifier", type=Path, default=ROOT / "tests/ns06_cross_slot_verify.py")
    parser.add_argument("--node", default="node")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--library-path", default="")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    frozen.require(os.environ.get("UMMINFER_LIVE_OPT_IN") == "1", "live opt-in required")
    frozen.require(len(args.execution_commit) == 40, "full execution commit required")
    frozen.require(not args.out_dir.exists(), "refusing to overwrite evidence")
    args.out_dir.mkdir(parents=True)
    prereg = frozen.load_prereg(args.preregistration)
    prereg["after_runtime_commit"] = args.execution_commit
    runtime = {name: getattr(args, name) for name in ("server", "vlm", "asr", "tts", "wan", "flux")}
    report = frozen.run_arm(args, prereg, "after", runtime, args.out_dir / "after")
    print(json.dumps({"status": "CAPTURED", "runtime_commit": report["runtime_commit"],
                      "images": sum(row["media_type"] == "image" for row in report["main"]["artifacts"]),
                      "videos": sum(row["media_type"] == "video" for row in report["main"]["artifacts"]),
                      "native_narration_legs": report["native_narration_legs"],
                      "narration_leg_target_met": report["narration_leg_target_met"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
