#!/usr/bin/env python3
"""Fresh-process byte-identical verification of a realtime latency audit."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "umminfer_realtime_latency_audit_fresh", ROOT / "python/umminfer_realtime_latency_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--profile", choices=("task41-r3", "generic"), default="task41-r3")
    args = parser.parse_args()
    try:
        expected = audit.render_report(audit.audit_evidence(args.evidence, args.profile))
        actual = args.report.read_bytes()
        if actual != expected.encode("utf-8"):
            raise audit.AuditError("fresh-process report is not byte-identical")
        parsed = json.loads(actual)
        digest = hashlib.sha256(actual).hexdigest()
    except (audit.AuditError, OSError, ValueError, TypeError, KeyError) as error:
        print(f"REALTIME_LATENCY_INDEPENDENT FAIL error={error}", file=sys.stderr)
        return 2
    print("REALTIME_LATENCY_INDEPENDENT PASS "
          f"connections={parsed['counts']['connections']} "
          f"connection_response_ids={parsed['counts']['connection_scoped_response_identities']} "
          f"server_responses={parsed['counts']['server_responses']} "
          f"prompts={parsed['counts']['native_vlm_prompts']} "
          f"terminals={parsed['counts']['terminal_callbacks']} sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
