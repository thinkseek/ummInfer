#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent fresh-process audit of a real online-rollout evidence bundle."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import sse_replay_live as replay_tools  # noqa: E402
import umminfer_online_rollout as online  # noqa: E402


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(
        encoding="utf-8").splitlines() if line]


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("usage: online_rollout_live_verify.py /absolute/evidence-dir")
    root = Path(args[0])
    require(root.is_absolute() and root.is_dir(), "evidence directory must be absolute")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    require(summary["status"] == "PASS", "live gate did not pass")
    require(summary["plan"] == {"groups": 2, "seeds_per_group": 3},
            "frozen plan shape drifted")
    require(summary["health_before_shutdown"]["in_flight"] == 0,
            "shutdown health was not in_flight=0")
    for key in ("gpu7_before", "gpu7_after_serving", "gpu7_final"):
        require(summary[key]["index"] == 7 and
                summary[key]["memory_used_mib"] <= 16 and
                not summary[key]["compute_processes"], f"{key} was not idle")
    require(summary["claims"]["optimizer"] == "NOT_RUN" and
            summary["claims"]["model_quality"] == "NOT_RUN",
            "evidence contains an optimizer or quality claim")

    identity = summary["native_identity"]
    require(identity["schema"] == "umminfer-native-stage-identity-v2" and
            identity["state"] == "loaded" and
            identity["stage_config"]["batch_size"] == 1,
            "real identity is not a loaded serial v2 native stage")
    model_root = Path(identity["model_root"])
    config_bytes = (model_root / identity["config"]["path"]).read_bytes()
    require(len(config_bytes) == identity["config"]["bytes"] and
            hashlib.sha256(config_bytes).hexdigest() == identity["config"]["sha256"],
            "independent model-config rehash differs")
    actual_weights = {}
    for relative, declared in identity["weight_files"].items():
        path = model_root / relative
        require(path.is_file() and not path.is_symlink(),
                f"weight is missing or a symlink: {relative}")
        measured = replay_tools.sha256_file(path)
        require(measured == declared, f"independent weight rehash differs: {relative}")
        actual_weights[relative] = measured
    weight_set = online.sha256_bytes(online.canonical_json(actual_weights))
    require(weight_set == identity["model_sha256"], "independent weight-set hash differs")
    require(summary["provenance"]["model_sha256"] == weight_set and
            summary["provenance"]["config_sha256"] == identity["config"]["sha256"],
            "recorded model provenance differs from independent hashes")

    plan_bytes = (root / "config/plan.json").read_bytes()
    plan = online.prepare_plan(json.loads(plan_bytes.decode("utf-8")))
    require(plan_bytes == online._canonical_line(plan), "plan bytes are not canonical")
    require(plan["behavior_policy"]["model_sha256"] == weight_set and
            plan["behavior_policy"]["model_config_sha256"] == identity["config"]["sha256"],
            "plan is not bound to independently verified model identity")
    schedule, schedule_sha = online.build_schedule(plan)
    accepted = root / "accepted"
    records = read_jsonl(accepted / "trajectories.jsonl")
    require(len(records) == len(schedule) == 6, "accepted records are not 2x3")
    by_rollout = {record["rollout_provenance"]["rollout_id"]: record
                  for record in records}
    require(set(by_rollout) == {item["rollout_id"] for item in schedule},
            "accepted rollout identities are stale or incomplete")
    for item in schedule:
        record = by_rollout[item["rollout_id"]]
        online.validate_online_record(
            record, item, plan, schedule_sha, accepted / "artifacts")
        provenance = record["rollout_provenance"]
        require(provenance["response"]["behavior_evidence"] == "exposed" and
                provenance["behavior"]["status"] == "COVERED" and
                provenance["behavior"]["token_count"] > 0,
                f"accepted rollout is not exposed: {item['rollout_id']}")
    require(len({record["request_id"] for record in records}) == 6 and
            len({record["session_id"] for record in records}) == 6,
            "request/session identities are not unique")
    attempts = read_jsonl(accepted / "attempts.jsonl")
    statuses = [(record.get("rollout_provenance") or {}).get(
        "response", {}).get("http_status") for record in attempts]
    require(503 in statuses and statuses[-1] == 200,
            "bounded retry attempts are not preserved")
    require(len(attempts) > 6 and len(attempts) <= 6 + 4,
            "accepted attempts exceeded the frozen retry bound")

    negative = summary["negative_cases"]
    require({case["case"] for case in negative} ==
            {"default", "truncated", "cancelled", "unsupported-video"},
            "negative coverage set is incomplete")
    require(all(case["coverage"] == "not_exposed" and
                case["accepted_into_group"] is False for case in negative),
            "a negative response entered an accepted group")
    forwards = summary["fresh_forward"]
    require(len(forwards) == 6 and
            all(item["sampled_ids_exact"] and item["mask_exact"] and
                item["clean_text_exact"] and
                item["max_abs_logprob_diff"] <= item["abs_tolerance"] <= 1e-6
                for item in forwards),
            "independent native numerical gate is incomplete")

    independent = root / "independent-replay"
    online.replay_bundle(accepted, independent)
    for relative in online._accepted_relative_files(accepted):
        require((accepted / relative).read_bytes() == (independent / relative).read_bytes(),
                f"independent replay differs: {relative}")
    result = {
        "schema": "umminfer-online-rollout-independent-verification-v1",
        "status": "PASS", "groups": 2, "rollouts": 6,
        "accepted_attempts": len(attempts), "http_attempt_statuses": statuses,
        "negative_cases": len(negative), "fresh_forward_cases": len(forwards),
        "model_files_rehashed": len(actual_weights) + 1,
        "weight_set_sha256": weight_set,
        "byte_identical_fresh_process_replay": True,
        "accepted_files_compared": len(online._accepted_relative_files(accepted)),
        "optimizer": "NOT_RUN", "model_quality": "NOT_RUN",
    }
    (root / "independent-verification.json").write_bytes(
        online.canonical_json(result) + b"\n")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            files.append(f"{replay_tools.sha256_file(path)['sha256']}  "
                         f"{path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(files) + "\n", encoding="ascii")
    checked = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=root,
                             capture_output=True, text=True, check=False)
    require(checked.returncode == 0,
            f"SHA256SUMS verification failed: {checked.stdout}{checked.stderr}")
    print("ONLINE_ROLLOUT_INDEPENDENT_VERIFY: PASS "
          f"groups=2 rollouts=6 weights={len(actual_weights)} "
          f"files={len(files)} byte_identical=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
