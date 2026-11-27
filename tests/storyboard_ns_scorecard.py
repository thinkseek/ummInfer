#!/usr/bin/env python3
"""Build a compact paired NS-06/NS-08 scorecard from external raw browser evidence."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha(path):
    return sha(Path(path).read_bytes())


def first_sse(result, root):
    entries = [x for x in result["requests"] if x.get("content_type") == "text/event-stream" and x.get("file")]
    expected_slots = result.get("initial", {}).get("slots")
    for entry in entries:
        path = Path(root) / "network" / entry["file"]
        kinds = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].lstrip(" ")
            if payload == "[DONE]":
                continue
            unified = json.loads(payload).get("unified")
            if unified:
                kinds.append(unified.get("kind"))
        if "asr_final" in kinds and kinds.count("visual_slot") == expected_slots:
            return path
    raise ValueError("no raw SSE stream matches the initial ASR final and observed visual-slot count")


def raw_summary(result, root):
    path = first_sse(result, root)
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].lstrip(" ")
        if payload == "[DONE]":
            continue
        unified = json.loads(payload).get("unified")
        if unified:
            events.append(unified)
    terminal = [x for x in events if x.get("kind") in ("complete", "cancelled", "error")]
    complete_text = "".join(x.get("text", "") for x in events if x.get("kind") == "complete")
    asr_text = next((x.get("text", "") for x in events if x.get("kind") == "asr_final"), "")
    labels = sorted(set(re.findall(r"(?:^|\n)\s*#{0,3}\s*shot\s+([123])(?:\s|:|#)", complete_text, re.I)))
    return {
        "sha256": file_sha(path), "events": len(events), "terminals": len(terminal),
        "script_labels_detected": len(labels), "script_labels": labels,
        "asr_text": asr_text, "clean_without_directives": "[[" not in complete_text,
        "pcm_artifacts": sum(x.get("kind") == "artifact" and
                             x.get("artifact", {}).get("mime_type") == "audio/pcm" for x in events),
    }


def checks(result):
    return {x["name"]: x for x in result["checks"]}


def passed(result, name):
    item = checks(result).get(name)
    return bool(item and item["ok"])


def omitted_count(result):
    item = checks(result).get("exact media restored by After; Before omission is explicit")
    return item.get("detail", {}).get("omitted") if item else None


def source_binding(result, archive):
    if not archive.get("turns"):
        return None
    first_slots = archive["turns"][0].get("visual_slots", [])
    if not first_slots:
        return None
    slot = first_slots[0]
    original = next((x.get("sha256") for x in slot.get("variants", []) if x.get("kind") == "image"),
                    slot.get("sha256"))
    candidates = [x for x in result["requests"] if x.get("image_edit_sha256")]
    return {
        "original_sha256": original,
        "requests": [{"image_input_sha256": x.get("image_input_sha256"),
                      "image_edit_sha256": x.get("image_edit_sha256")} for x in candidates],
        "exact": bool(candidates) and all(x.get("image_input_sha256") == original and
                                          x.get("image_edit_sha256") == original for x in candidates),
    }


def zero_manual_controls(result):
    forbidden_actions = [x for x in result["actions"] if any(word in x["name"].lower()
                         for word in ("preset", "frame budget", "modality", "sampling", "seed field"))]
    forbidden_requests = [x for x in result["requests"] if x.get("seed_present") or x.get("frame_selection_present")]
    return not forbidden_actions and not forbidden_requests


def arm_summary(result, root, archive, stack):
    raw = raw_summary(result, root)
    archive_path = Path(root) / "session.json"
    binding = source_binding(result, archive)
    return {
        "checks_passed": sum(x["ok"] for x in result["checks"]),
        "checks_failed": len(result["failures"]),
        "failures": result["failures"],
        "initial_keyframes": result.get("initial", {}).get("images", 0),
        "initial_video_slots": result.get("initial", {}).get("videos", 0),
        "i2v_previews_after_animate": result.get("after_animate", {}).get("videos", 0),
        "script_labels_detected": raw["script_labels_detected"],
        "native_pcm_artifacts": raw["pcm_artifacts"],
        "asr_text_sha256": sha(raw["asr_text"].encode()),
        "asr_required_terms": {x: x in raw["asr_text"].lower() for x in ("three", "cafe", "watch", "smile")},
        "clean_without_directives": raw["clean_without_directives"],
        "unique_initial_terminal": raw["terminals"] == 1,
        "raw_sse_sha256": raw["sha256"],
        "revision_supported": result.get("revision_supported", False),
        "revision_produces_visual": passed(result, "revision produces a new visual artifact"),
        "source_binding": binding,
        "cancel_no_stale": passed(result, "cancel terminal with no stale DOM/audio update"),
        "cancel_no_shimmer": passed(result, "cancel leaves no unresolved shimmer"),
        "retry_completed": passed(result, "natural retry completes"),
        "export": result.get("export"),
        "export_file_sha256": file_sha(archive_path),
        "exact_text_history": passed(result, "exact text/history restored"),
        "media_omitted_after_import": omitted_count(result),
        "reconnect_preserved_history": passed(result, "reconnect preserves imported text/history"),
        "followup_uses_history": passed(result, "follow-up uses restored session history"),
        "zero_manual_modality_or_frame_controls": zero_manual_controls(result),
        "advanced_initially_open": result.get("advanced_open"),
        "developer_control_visibility": result.get("developer_control_visibility"),
        "stack_execution_commit": stack.get("execution_commit"),
        "stack_report_sha256": None,
    }


def main():
    p = argparse.ArgumentParser()
    for name in ("preregistration", "before", "after", "before-stack", "after-stack", "after-verification", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    args = p.parse_args()
    prereg = load(args.preregistration)
    before = load(args.before / "result.json")
    after = load(args.after / "result.json")
    before_stack = load(args.before_stack)
    after_stack = load(args.after_stack)
    after_verification = load(args.after_verification)
    before_archive = load(args.before / "session.json")
    after_archive = load(args.after / "session.json")
    b = arm_summary(before, args.before, before_archive, before_stack)
    a = arm_summary(after, args.after, after_archive, after_stack)
    b["stack_report_sha256"] = file_sha(args.before_stack)
    a["stack_report_sha256"] = file_sha(args.after_stack)
    shared_binaries = before_stack.get("binaries") == after_stack.get("binaries")
    shared_models = before_stack.get("models") == after_stack.get("models")
    ns06_engineering = (
        a["initial_keyframes"] == 3 and a["i2v_previews_after_animate"] == 3 and
        a["script_labels_detected"] == 3 and a["native_pcm_artifacts"] > 0 and
        a["clean_without_directives"] and all(a["asr_required_terms"].values()) and
        a["unique_initial_terminal"] and a["source_binding"] and a["source_binding"]["exact"] and
        a["revision_produces_visual"] and
        a["cancel_no_stale"] and a["cancel_no_shimmer"] and a["retry_completed"] and
        a["zero_manual_modality_or_frame_controls"]
    )
    ns08_engineering = (
        after_archive.get("schema") == "umminfer-web-session-replay-v1" and
        a["exact_text_history"] and a["media_omitted_after_import"] == 0 and
        a["reconnect_preserved_history"] and a["followup_uses_history"] and
        a["zero_manual_modality_or_frame_controls"]
    )
    no_regression = (
        a["initial_keyframes"] >= b["initial_keyframes"] and
        a["i2v_previews_after_animate"] >= b["i2v_previews_after_animate"] and
        a["script_labels_detected"] >= b["script_labels_detected"] and
        a["cancel_no_stale"] and a["cancel_no_shimmer"] and a["retry_completed"]
    )
    source_matches = {}
    for side, commit in (("before", prereg["before"]), ("after", prereg["after"])):
        raw = subprocess.check_output(["git", "show", f"{commit['runtime_commit']}:web/index.html"])
        source_matches[side] = sha(raw) == commit["web_sha256"]
    raw_verified = (after_verification.get("status") == "PASS" and
                    after_verification.get("archive_sha256") == a["export_file_sha256"])
    paired_improvement = ns06_engineering and ns08_engineering and no_regression and raw_verified and shared_binaries and shared_models and all(source_matches.values())
    report = {
        "schema": "umminfer-ns06-ns08-paired-scorecard/v1",
        "preregistration_sha256": file_sha(args.preregistration),
        "source_commits": {"before": prereg["before"]["runtime_commit"], "after": prereg["after"]["runtime_commit"]},
        "source_hash_matches": source_matches,
        "shared_native_binaries": shared_binaries,
        "shared_native_binary_sha256": {k: v.get("sha256") for k, v in before_stack.get("binaries", {}).items()},
        "shared_model_identities": shared_models,
        "shared_model_tree_sha256": {k: v.get("tree_sha256") for k, v in before_stack.get("models", {}).items()},
        "before": b,
        "after": a,
        "after_independent_raw_verification": {
            "status": after_verification.get("status"),
            "sha256": file_sha(args.after_verification),
            "archive_hash_matches": raw_verified,
        },
        "engineering_gates": {
            "ns06_counts_revision_cancel_retry": "MET" if ns06_engineering else "NOT_MET",
            "ns08_default_export_import_reconnect": "MET" if ns08_engineering else "NOT_MET",
            "no_after_regression": "MET" if no_regression else "NOT_MET",
            "paired_raw_binary_model_identity": "MET" if raw_verified and shared_binaries and shared_models else "NOT_MET",
        },
        "paired_outcome": "PAIRED_ENGINEERING_IMPROVEMENT_ONLY" if paired_improvement else "NOT_PROMOTED",
        "north_star_hard_gates": {
            "NS-06": "NOT_MET — human character/style consistency NOT_RUN",
            "NS-08": "NOT_RUN — macOS and human walkthrough NOT_RUN",
        },
        "scoreboard_update_allowed": False,
        "deployment_or_resident_readiness": "NOT_CLAIMED",
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"paired_outcome": report["paired_outcome"],
                      "engineering_gates": report["engineering_gates"],
                      "north_star_hard_gates": report["north_star_hard_gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
