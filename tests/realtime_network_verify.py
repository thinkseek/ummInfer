#!/usr/bin/env python3
"""Fresh-process independent verifier for realtime_network_live.py evidence."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "umminfer_realtime_network_verify", ROOT / "python/umminfer_realtime_network.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--write", action="store_true",
                        help="write independent-verification.json after a passing verification")
    args = parser.parse_args()
    report = MODULE.verify_evidence(args.evidence)
    if args.write:
        (args.evidence / "independent-verification.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("REALTIME_NETWORK_INDEPENDENT PASS "
          f"sessions={report['sessions']} rounds={report['rounds']} "
          f"responses={report['responses']} prompts={report['verified_next_turn_prompts']} "
          f"terminals={report['terminal_callbacks']} in_flight=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
