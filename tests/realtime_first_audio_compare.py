#!/usr/bin/env python3
"""Fresh-process controlled A/B comparison for two sealed first-audio reports."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "umminfer_realtime_first_audio_compare_module", ROOT / "python/umminfer_realtime_first_audio.py")
first = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(first)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise first.FirstAudioError("refusing to overwrite an existing A/B report")
        baseline = json.loads(args.baseline.read_text())
        candidate = json.loads(args.candidate.read_text())
        result = first.compare_reports(baseline, candidate)
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        args.output.write_text(rendered)
    except (first.FirstAudioError, OSError, ValueError, TypeError, KeyError) as error:
        print(f"REALTIME_FIRST_AUDIO_AB FAIL error={error}", file=sys.stderr)
        return 2
    print(f"REALTIME_FIRST_AUDIO_AB {result['status']} pairs={result['pairs']} "
          f"median_gain={result['primary']['median_gain_fraction']:.6f} "
          f"p95_change={result['primary']['p95_change_fraction']:.6f} "
          f"serial_change={result['serial_guardrail']['median_change_fraction']:.6f}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
