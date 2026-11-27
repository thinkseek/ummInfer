#!/usr/bin/env python3
"""Independent fresh-process verification of one sealed NS-08 arm bundle."""

import argparse
import hashlib
import json
from pathlib import Path

GROUPS = {
    "visible_modality_control_count": {"image", "audio", "video"},
    "visible_routing_control_count": {"narrative", "visualMode", "maxSlots", "allowTextOnly",
        "narrativeContinuity", "styleAnchor", "fastVideo"},
    "visible_sampling_control_count": {"maxTokens", "temperature", "topP", "seed",
        "imageSteps", "videoSteps", "videoGuidance"},
    "visible_frame_control_count": {"videoFramePolicy", "videoFrameBudget",
        "videoMotionThreshold", "videoSceneThreshold"},
    "visible_voice_routing_control_count": {"voiceToggle", "voice", "realtimeEchoSuppression"},
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.arm_dir.resolve()
    arm_path = root / "NS-08.json"
    browser_path = root / "browser-result.json"
    manifest_path = root / "SHA256SUMS"
    arm = json.loads(arm_path.read_text(encoding="utf-8"))
    browser = json.loads(browser_path.read_text(encoding="utf-8"))
    require(arm.get("case_id") == "NS-08" and
            arm.get("schema") == "umminfer-ns08-zero-control-arm/v1", "wrong arm identity")
    require(arm.get("full_gate_status") == "NOT_RUN" and
            arm.get("production_status") == "NOT_RUN", "arm cannot claim full NS-08")
    require(arm.get("coverage", {}).get("macos_real") == "NOT_RUN" and
            arm.get("coverage", {}).get("human_walkthrough") == "NOT_RUN",
            "arm lacks explicit macOS/human NOT_RUN coverage")
    commit = arm.get("source_identity", {}).get("commit", "")
    require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit),
            "arm lacks exact source commit")
    visible = set(browser.get("visible_controls", []))
    require(len(visible) == len(browser.get("visible_controls", [])),
            "duplicate visible-control identities")
    for metric, controls in GROUPS.items():
        require(arm["metrics"][metric] == len(visible & controls) ==
                browser["visible_counts"][metric], f"visibility count mismatch: {metric}")
    failed = [row for row in browser.get("checks", []) if not row.get("ok")]
    require(browser.get("checks_failed") == len(failed) == arm["metrics"]["browser_checks_failed"],
            "browser failure count mismatch")
    require(arm["raw_evidence"].get("browser_result_sha256") == sha(browser_path),
            "browser result digest is not bound")
    if arm.get("engineering_status") == "CONTRACT_PASS":
        require(browser.get("status") == "PASS" and not visible and not failed,
                "engineering PASS lacks a zero-control browser PASS")
        require(arm["metrics"]["contract_checks_failed"] == 0,
                "engineering PASS retains contract failures")
        require(all(arm["metrics"][name] == 1 for name in
                    ("export_import_exact", "reconnect_exact", "source_bound_revision",
                     "cancellation_retry_exact")),
                "engineering PASS dropped an accepted Task #77 behavior")
    else:
        require(arm.get("engineering_status") == "FAIL", "unexpected arm engineering state")
    listed = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        require(separator and len(digest) == 64, "malformed manifest row")
        require(name != "SHA256SUMS" and not Path(name).is_absolute() and
                ".." not in Path(name).parts and name not in listed, "unsafe/duplicate manifest row")
        listed[name] = digest
    actual = {path.relative_to(root).as_posix(): path for path in root.rglob("*")
              if path.is_file() and path != manifest_path}
    require(set(actual) == set(listed), "NS-08 manifest is not complete and self-excluded")
    for name, path in actual.items():
        require(sha(path) == listed[name], f"NS-08 evidence digest mismatch: {name}")
    report = {
        "schema": "umminfer-ns08-arm-independent-verification/v1",
        "status": "PASS", "arm": arm.get("arm"),
        "source_commit": commit, "engineering_status": arm["engineering_status"],
        "visible_developer_controls": len(visible),
        "browser_checks_failed": len(failed),
        "arm_sha256": sha(arm_path), "manifest_sha256": sha(manifest_path),
        "manifest_file_count": len(listed), "verifier_sha256": sha(Path(__file__).resolve()),
        "producer_imported": False, "macos_real": "NOT_RUN", "human_walkthrough": "NOT_RUN",
    }
    if args.out.exists():
        raise SystemExit("refusing to overwrite an existing verification report")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
