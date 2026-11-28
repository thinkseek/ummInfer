#!/usr/bin/env python3
"""Independent verification of the retained NS-05 NOT_MET co-creation arms."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import ns05_voice_cocreation_verify as raw  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_arm(root):
    root = Path(root).resolve()
    prereg = raw.load(root / "preregistration.json")
    requests = raw.load(root / "requests.json")
    require(raw.sha(root / "source.mp4") == prereg["source"]["derivative_sha256"], "source")
    for name, expected in prereg["voice_cocreation_journey"]["input_audio_sha256"].items():
        require(raw.sha(root / f"{name}.wav") == expected, f"voice: {name}")
    expected = {"phone": "complete", "scene": "cancelled",
                "cyberpunk-initial": "cancelled", "cyberpunk-retry": "complete"}
    records = {label: raw.verify_public(root, label, terminal)
               for label, terminal in expected.items()}
    require(len({row["request_id"] for row in records.values()}) == 4, "response identity")
    require(records["cyberpunk-initial"]["request_sha256"] ==
            records["cyberpunk-retry"]["request_sha256"], "retry body")
    require(all(row["asr_transcripts"] for row in records.values()), "ASR coverage")
    retry = records["cyberpunk-retry"]
    require(not [row for row in retry["artifacts"] if row["media_type"] in ("image", "video")],
            "failed self-reference unexpectedly generated visual media")
    slots = retry["visual_slots"]
    require(len(slots) == 1 and slots[0]["slot_id"] == "slot-0"
            and slots[0]["media_type"] == "video" and slots[0]["anchor_explicit"] is True,
            "frozen self-reference failure shape")
    visual_legs = [leg for leg in retry["terminal"]["legs"] if leg.get("stage") == "video"]
    require(len(visual_legs) == 1 and visual_legs[0]["state"] == "failed"
            and visual_legs[0]["error"] == "source slot slot-0 is not an earlier slot in this request",
            "self-reference was not rejected fail-closed")
    import msgpack
    submits, outputs, wan = [], [], []
    for path in sorted(root.glob("*.msgpack"), key=raw.tap_ordinal):
        message = msgpack.unpackb(path.read_bytes(), raw=False)
        if path.name.startswith("tts-output-") and message.get("type") in ("stream", "complete"):
            outputs.append(message)
        elif path.name.startswith("tts-") and message.get("type") == "submit":
            submits.append(message)
        elif path.name.startswith("video-") and message.get("type") == "submit":
            wan.append(message)
    require(not wan, "self-reference reached native Wan")
    joins = {label: raw.verify_pcm(root, records[label], submits, outputs)
             for label in ("phone", "cyberpunk-retry")}
    cleanup = raw.load(root / "cleanup.json")
    require(cleanup["post"]["memory_used_mib"] <= 16
            and cleanup["in_flight"]["in_flight"] == 0, "cleanup")
    return {"status": "NOT_MET", "reason": "MODEL_EMITTED_VIDEO_SLOT0_SOURCE_SLOT0_SELF_REFERENCE",
        "request_ids": {label: row["request_id"] for label, row in records.items()},
        "asr_transcripts": {label: row["asr_transcripts"] for label, row in records.items()},
        "native_pcm_joins": joins, "image_artifacts": 0, "video_artifacts": 0,
        "native_wan_submits": 0, "self_reference_fail_closed": True,
        "cancel_retry_exact": True, "stale_or_cross_session_events": 0, "final_in_flight": 0,
        "physical_acoustic_and_perceptual": "NOT_RUN"}


def verify_manifest(root):
    root = Path(root).resolve()
    expected = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = PurePosixPath(relative)
        require(not path.is_absolute() and ".." not in path.parts and relative != "SHA256SUMS",
                "unsafe manifest")
        expected[relative] = digest
    actual = {path.relative_to(root).as_posix(): path for path in root.rglob("*")
              if path.is_file() and path.name != "SHA256SUMS"}
    require(set(actual) == set(expected), "manifest file set")
    for relative, path in actual.items():
        require(raw.sha(path) == expected[relative], f"manifest: {relative}")
    return {"status": "PASS", "files": len(actual),
            "bytes": sum(path.stat().st_size for path in actual.values()),
            "sha256sums_sha256": raw.sha(root / "SHA256SUMS")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.manifest_only:
            require(len(args.root) == 1, "one manifest root")
            result = verify_manifest(args.root[0])
        else:
            require(len(args.root) == 2, "Before and After roots required")
            result = {"status": "NOT_MET", "before": verify_arm(args.root[0]),
                      "after": verify_arm(args.root[1]),
                      "promotion": "NOT_PROMOTED", "visual_cocreation_completed": False}
        Path(args.output).write_bytes(json.dumps(result, indent=2, sort_keys=True,
            ensure_ascii=False).encode() + b"\n")
        print(json.dumps({"status": result["status"], "output": args.output}, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"NS05 NEGATIVE VERIFY FAIL: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
