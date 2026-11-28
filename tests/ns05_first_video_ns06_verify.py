#!/usr/bin/env python3
"""Independently compare a new NS-06 After arm with accepted raw After evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import ns06_cross_slot_verify as raw  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path,
                        default=ROOT / "benchmarks/cross-slot-visual-continuity/ns06-preregistration.json")
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--reconstruction-dir", type=Path, required=True)
    parser.add_argument("--accepted-reconstruction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--node", default="node")
    args = parser.parse_args()
    try:
        raw.require(len(args.base_commit) == len(args.execution_commit) == 40, "full commit ids")
        args.reconstruction_dir.mkdir()
        args.accepted_reconstruction_dir.mkdir()
        prereg_sha = hashlib.sha256(args.preregistration.read_bytes()).hexdigest()
        original = raw.load_prereg(prereg_sha)
        candidate = dict(original)
        candidate["after_runtime_commit"] = args.execution_commit
        new = raw.verify_arm(args.root, "after", candidate, args.reconstruction_dir,
                             args.ffmpeg, args.ffprobe, args.node)
        new_provenance = raw.verify_provenance(args.root, "after", args.node)
        old = raw.verify_arm(args.accepted_root, "after", original,
                             args.accepted_reconstruction_dir, args.ffmpeg, args.ffprobe, args.node)
        old_provenance = raw.verify_provenance(args.accepted_root, "after", args.node)
        for field in ("clean_text_sha256", "asr_final_sha256", "visual_states"):
            raw.require(new[field] == old[field], f"NS06 identity changed: {field}")
        raw.require(new["tts"]["pcm_sha256"] == old["tts"]["pcm_sha256"]
                    and new["tts"]["text_sha256"] == old["tts"]["text_sha256"]
                    and new["tts"]["pcm_bytes"] == old["tts"]["pcm_bytes"],
                    "NS06 PCM/text join changed")
        raw.require(new["chain"] == old["chain"], "NS06 exact chain hashes changed")
        raw.require(new["media"] == old["media"], "NS06 decoded media/continuity changed")
        result = {
            "schema_version": 1,
            "experiment": "umminfer_ns05_first_video_ns06_non_regression_v1",
            "status": "PASS", "base_commit": args.base_commit,
            "execution_commit": args.execution_commit,
            "accepted_task76_execution": original["after_runtime_commit"],
            "preregistration_sha256": prereg_sha,
            "keyframe_count": 3, "i2v_count": 3,
            "exact_image_edit_links": 2, "exact_i2v_links": 3,
            "clean_text_pcm_asr_media_identical_to_accepted_after": True,
            "candidate": new, "accepted_after": old,
            "candidate_provenance": {"binaries": new_provenance["binaries"],
                "model_tree_sha256": {name: model["tree_sha256"] for name, model in new_provenance["models"].items()}},
            "accepted_provenance": {"binaries": old_provenance["binaries"],
                "model_tree_sha256": {name: model["tree_sha256"] for name, model in old_provenance["models"].items()}},
            "human_perceptual_character_style": "NOT_RUN",
            "ns06_narration_leg_target": {
                "expected": 3, "candidate": new["tts"]["native_leg_count"],
                "accepted": old["tts"]["native_leg_count"],
                "met": new["tts"]["native_leg_target_met"] and old["tts"]["native_leg_target_met"],
            },
            "final_in_flight": 0,
        }
        args.output.write_bytes(json.dumps(result, indent=2, sort_keys=True,
            ensure_ascii=False, allow_nan=False).encode() + b"\n")
        print(json.dumps({"status": "PASS",
                          "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                          "clean_text_sha256": new["clean_text_sha256"],
                          "pcm_sha256": new["tts"]["pcm_sha256"],
                          "narration_leg_target_met": result["ns06_narration_leg_target"]["met"]}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, AssertionError) as error:
        print(f"NS06 FOLLOWUP VERIFY FAIL: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
