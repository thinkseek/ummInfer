#!/usr/bin/env python3
"""Fresh-process independent verification of a realtime first-audio evidence tree."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "umminfer_realtime_first_audio_fresh", ROOT / "python/umminfer_realtime_first_audio.py")
first = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(first)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--write", action="store_true", help="write report.json before SHA256SUMS sealing")
    args = parser.parse_args()
    try:
        report_path = args.evidence / "report.json"
        sealed = (args.evidence / "SHA256SUMS").exists()
        if args.write:
            if sealed or report_path.exists():
                raise first.FirstAudioError("refusing to overwrite a sealed/existing report")
        else:
            if not sealed:
                raise first.FirstAudioError("fresh verification requires SHA256SUMS")
        report = first.analyze(args.evidence, verify_sealed=not args.write)
        rendered = first.render_report(report).encode()
        if args.write:
            temporary = args.evidence / ".report.json.tmp"
            temporary.write_bytes(rendered)
            temporary.replace(report_path)
        else:
            actual = report_path.read_bytes()
            if actual != rendered:
                raise first.FirstAudioError("fresh-process report is not byte-identical")
        digest = hashlib.sha256(rendered).hexdigest()
    except (first.FirstAudioError, first.audit.AuditError, OSError, ValueError,
            TypeError, KeyError) as error:
        print(f"REALTIME_FIRST_AUDIO_INDEPENDENT FAIL error={error}", file=sys.stderr)
        return 2
    print(f"REALTIME_FIRST_AUDIO_INDEPENDENT PASS arms={report['counts']['arms']} "
          f"measured={report['counts']['measured_responses']} "
          f"responses={report['counts']['all_server_responses']} sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
