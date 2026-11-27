#!/usr/bin/env python3
"""Offline candidate/preflight/rollback identity check; never contacts a service."""

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED_ZERO_METRICS = (
    "unexpected_error_count", "process_crash_count", "cross_session_event_count",
    "stale_event_count", "duplicate_terminal_count", "pcm_mismatch_count", "final_in_flight",
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--after-arm", required=True, type=Path,
                        help="normalized NS-07.json from the isolated After run")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    after = json.loads(args.after_arm.read_text(encoding="utf-8"))
    failures = []

    def check(name, condition):
        if not condition:
            failures.append(name)

    check("candidate schema", candidate.get("schema") == "umminfer-ns07-upgrade-candidate/v1")
    check("After case", after.get("case_id") == "NS-07")
    check("isolated engineering gate", after.get("engineering_status") == "ISOLATED_PASS")
    check("production remains unrun", after.get("production_status") == "NOT_RUN")
    check("owner authorization absent", candidate.get("owner_rollout_authorization") is False)
    check("resident contact forbidden", candidate.get("resident_contact") == "FORBIDDEN")
    source = candidate.get("source", {})
    for name in ("before_commit", "after_commit"):
        value = source.get(name, "")
        check(f"exact {name}", len(value) == 40 and all(c in "0123456789abcdef" for c in value))
    check("source identity match", source.get("after_commit") ==
          after.get("source_identity", {}).get("commit"))
    check("binary identity match", candidate.get("binaries") == after.get("binary_identities"))
    check("model identity match", candidate.get("models") == after.get("model_identities"))
    check("config identity match", candidate.get("configuration") == after.get("config_identity"))
    metrics = after.get("metrics", {})
    check("full measured window", metrics.get("elapsed_s", 0) >= 1800)
    check("six logical clients", metrics.get("client_count") == 6)
    for name in REQUIRED_ZERO_METRICS:
        check(name, metrics.get(name) == 0)
    check("every fault recovered", metrics.get("recovery_attempt_count", 0) > 0 and
          metrics.get("recovery_attempt_count") == metrics.get("recovery_success_count"))
    check("no same-topology device growth", metrics.get("device_growth_mib", 1) <= 0)
    check("bounded host RSS growth", metrics.get("rss_growth_mib", 65) <= 64)
    invariants = candidate.get("required_health_invariants", [])
    aborts = candidate.get("abort_conditions", [])
    check("health invariants present", isinstance(invariants, list) and len(invariants) >= 6 and
          all(isinstance(item, str) and item for item in invariants))
    check("abort conditions present", isinstance(aborts, list) and len(aborts) >= 6 and
          all(isinstance(item, str) and item for item in aborts))
    rollback = candidate.get("rollback", {})
    check("rollback source bound", rollback.get("source_commit") == source.get("before_commit"))
    check("rollback remains dry-run", rollback.get("execution") == "NOT_RUN")
    report = {
        "schema": "umminfer-ns07-upgrade-preflight/v1",
        "status": "PASS" if not failures else "FAIL",
        "checks_failed": failures,
        "candidate_sha256": sha(args.candidate),
        "after_arm_sha256": sha(args.after_arm),
        "source_commits": {"before": source.get("before_commit"), "after": source.get("after_commit")},
        "resident_preflight": "STATIC_IDENTITY_AND_ISOLATED_HEALTH_ONLY",
        "resident_upgrade": "NOT_RUN",
        "resident_rollback": "NOT_RUN",
        "owner_authorization_required": True,
        "verifier_sha256": sha(Path(__file__).resolve()),
        "network_or_process_actions": 0,
    }
    if not args.out.exists():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        raise SystemExit("refusing to overwrite an existing preflight report")
    print(json.dumps(report, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
