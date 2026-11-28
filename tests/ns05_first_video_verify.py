#!/usr/bin/env python3
"""Independently verify and score the paired NS-05 first-video correction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import ns05_voice_cocreation_verify as raw  # noqa: E402
import ns05_voice_cocreation_negative_verify as negative  # noqa: E402

LABELS = ("phone", "scene", "cyberpunk-initial", "cyberpunk-retry")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_bytes())


def digest(raw_bytes):
    return hashlib.sha256(raw_bytes).hexdigest()


def png_dimensions(path):
    header = Path(path).read_bytes()[:24]
    require(len(header) == 24 and header[:8] == b"\x89PNG\r\n\x1a\n"
            and header[12:16] == b"IHDR", "source is not a PNG IHDR")
    return list(struct.unpack(">II", header[16:24]))


def verify_execution_identity(path):
    identity = raw.load(path)
    require(identity["physical_gpu"] == 7 and identity["model_revision"] == "UNKNOWN",
            "execution hardware/revision identity")
    compact = {"physical_gpu": 7, "model_revision": "UNKNOWN",
               "python_runtime": identity["python_runtime"], "binaries": {}, "sources": {}, "models": {}}
    for group in ("binaries", "sources"):
        for name, item in identity[group].items():
            path = Path(item["path"])
            require(raw.sha(path) == item["sha256"] and path.stat().st_size == item["bytes"],
                    f"{group} identity: {name}")
            compact[group][name] = {"bytes": item["bytes"], "sha256": item["sha256"]}
    for name, model in identity["models"].items():
        root = Path(model["root"])
        files = {}
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            relative = path.relative_to(root).as_posix()
            files[relative] = {"bytes": path.stat().st_size, "sha256": raw.sha(path)}
        require(files == model["files"], f"model files: {name}")
        tree = digest(raw.canonical(files)) if hasattr(raw, "canonical") else digest(
            json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
        require(tree == model["tree_sha256"], f"model tree: {name}")
        compact["models"][name] = {"model_id": model["model_id"], "files": len(files),
            "bytes": sum(item["bytes"] for item in files.values()), "tree_sha256": tree}
    require(compact["sources"]["baseline_prompt"]["sha256"]
            != compact["sources"]["candidate_prompt"]["sha256"], "director-only source delta")
    return compact


def elapsed(record, marker):
    require(record[marker] is not None, f"missing {marker}: {record['label']}")
    return (record[marker] - record["started_monotonic_ns"]) / 1e6


def audio_join(root, record):
    chunks = [row for row in record["artifacts"] if row["media_type"] == "audio"]
    payload = b"".join((root / row["file"]).read_bytes() for row in chunks)
    require(payload and any(payload), f"silent/empty public PCM: {record['label']}")
    return {"chunks": len(chunks), "bytes": len(payload), "sha256": digest(payload)}


def timings(root, barge):
    records = {label: raw.load(root / f"{label}.json") for label in LABELS}
    barge_record = raw.load(root / "barge.json")
    require(barge_record["barge_start_monotonic_ns"] == barge, "barge clock identity")
    retry = records["cyberpunk-retry"]
    return {
        "phone_first_text": elapsed(records["phone"], "first_text_monotonic_ns"),
        "phone_first_audio": elapsed(records["phone"], "first_audio_monotonic_ns"),
        "retry_first_text": elapsed(retry, "first_text_monotonic_ns"),
        "retry_first_audio": elapsed(retry, "first_audio_monotonic_ns"),
        "barge_to_scene_terminal": (records["scene"]["finished_monotonic_ns"] - barge) / 1e6,
        "barge_to_initial_terminal": (records["cyberpunk-initial"]["finished_monotonic_ns"] - barge) / 1e6,
        "full_barge_to_retry_first_audio": (retry["first_audio_monotonic_ns"] - barge) / 1e6,
    }


def public_summary(root):
    result = {}
    for label in LABELS:
        record = raw.load(root / f"{label}.json")
        result[label] = {
            "request_id": record["request_id"],
            "request_sha256": record["request_sha256"],
            "terminal": record["terminal"]["kind"],
            "text_sha256": record["text_sha256"],
            "text_bytes": len(record["text"].encode()),
            "asr_transcripts": record["asr_transcripts"],
            "raw_sse_sha256": record["raw_sse_sha256"],
            "public_pcm": audio_join(root, record) if label in ("phone", "cyberpunk-retry") else "CANCELLED_PREFIX_RETAINED_EXTERNALLY",
        }
    return result


def verify(before_root, after_root, ffmpeg, ffprobe, base_commit, execution_commit, identities):
    before_root, after_root = Path(before_root).resolve(), Path(after_root).resolve()
    require(len(base_commit) == len(execution_commit) == 40, "full commit ids")
    execution_identity = verify_execution_identity(identities)
    before_negative = negative.verify_arm(before_root)
    after_positive = raw.verify(after_root, ffmpeg, ffprobe)
    prereg = raw.load(after_root / "preregistration.json")
    frozen_requests = raw.load(after_root / "requests.json")
    cyberpunk_request = frozen_requests["cyberpunk"]
    before_public = public_summary(before_root)
    after_public = public_summary(after_root)
    for label in LABELS:
        require(before_public[label]["request_sha256"] == after_public[label]["request_sha256"],
                f"frozen request changed: {label}")
        require(before_public[label]["asr_transcripts"] == after_public[label]["asr_transcripts"],
                f"natural ASR transcript changed: {label}")
        require(before_public[label]["terminal"] == after_public[label]["terminal"],
                f"terminal lifecycle changed: {label}")
    before_retry = raw.load(before_root / "cyberpunk-retry.json")
    after_retry = raw.load(after_root / "cyberpunk-retry.json")
    before_slots = before_retry["visual_slots"]
    require(len(before_slots) == 1 and before_slots[0]["slot_id"] == "slot-0"
            and before_slots[0]["media_type"] == "video"
            and before_slots[0]["anchor_explicit"] is True,
            "Before is not the frozen first-video self-reference")
    after_slots = after_retry["visual_slots"]
    require(len(after_slots) == 2, "After directive count")
    require(after_slots[0]["slot_id"] == "slot-0" and after_slots[0]["media_type"] == "image"
            and after_slots[0].get("anchor_explicit", False) is False,
            "After must establish unconditioned image slot-0 first")
    require(after_slots[1]["slot_id"] == "slot-1" and after_slots[1]["media_type"] == "video"
            and after_slots[1]["anchor_explicit"] is True
            and after_slots[1]["anchor_slot_index"] == 0,
            "After must animate exact earlier image slot-0 in video slot-1")
    images = [row for row in after_retry["artifacts"] if row["media_type"] == "image"]
    videos = [row for row in after_retry["artifacts"] if row["media_type"] == "video"]
    require(len(images) == len(videos) == 1, "After media cardinality")
    image, video = images[0], videos[0]
    require(image["slot_id"] == "slot-0" and video["slot_id"] == "slot-1"
            and video["anchor_slot_index"] == 0 and video["anchor_explicit"] is True
            and video["anchor_sha256"] == image["sha256"], "public exact source identity")
    dimensions = png_dimensions(after_root / image["file"])
    require(dimensions == [512, 512], "decoded FLUX dimensions")
    report = raw.load(after_root / "report.json")
    native_submit = report["native_video_submit"]
    require(native_submit["request_id"] == after_retry["request_id"] + ":slot:1"
            and native_submit["anchor_slot_index"] == 0
            and native_submit["anchor_explicit"] is True
            and native_submit["anchor_crop"] == "center_cover"
            and native_submit["image_sha256"] == image["sha256"]
            and native_submit["anchor_sha256"] == image["sha256"]
            and native_submit["image_bytes"] == image["bytes"], "native submit identity")
    packed_name = native_submit["packed_file"]
    require(Path(packed_name).name == packed_name
            and raw.sha(after_root / packed_name) == native_submit["packed_sha256"],
            "native MessagePack submit hash")
    stream = report["decoded_video"]
    require([stream["width"], stream["height"]] == [704, 480], "decoded Wan dimensions")
    require(report["execution_commit"] == execution_commit, "execution commit identity")
    before_barge = raw.load(before_root / "barge.json")["barge_start_monotonic_ns"]
    after_barge = raw.load(after_root / "barge.json")["barge_start_monotonic_ns"]
    before_times, after_times = timings(before_root, before_barge), timings(after_root, after_barge)
    first_response_pass = all(times[name] < 3000.0 for times in (before_times, after_times)
                              for name in ("phone_first_text", "phone_first_audio"))
    require(first_response_pass, "initial real-video first text/voice exceeded 3000 ms")
    require(before_negative["image_artifacts"] == before_negative["video_artifacts"]
            == before_negative["native_wan_submits"] == 0, "Before failure cardinality")
    require(after_positive["native_wan_source_exact"] and after_positive["cancel_retry_exact"]
            and after_positive["final_in_flight"] == 0, "After completion/lifecycle")
    promoted = bool(first_response_pass and after_positive["first_frame_mae"] <= 48.0
                    and after_positive["motion_mae"] >= 1.0)
    return {
        "schema_version": 1,
        "experiment": "umminfer_ns05_first_video_self_reference_v1",
        "status": "PASS" if promoted else "NOT_MET",
        "decision": "PROMOTED_CORRECTNESS" if promoted else "NOT_PROMOTED",
        "base_commit": base_commit,
        "execution_commit": execution_commit,
        "execution_identity": execution_identity,
        "frozen_inputs": {
            "preregistration_sha256": raw.sha(after_root / "preregistration.json"),
            "source_video": prereg["source"],
            "voice_wav_sha256": prereg["voice_cocreation_journey"]["input_audio_sha256"],
            "sampling": {key: cyberpunk_request[key] for key in
                         ("seed", "temperature", "top_p", "max_tokens")},
            "modalities": cyberpunk_request["modalities"],
            "image": cyberpunk_request["image"], "audio": cyberpunk_request["audio"],
            "video": cyberpunk_request["video"], "narrative": cyberpunk_request["narrative"],
            "client_directives": "NONE",
            "frame_selection": "ABSENT",
        },
        "physical_gpu": 7,
        "before": {
            "outcome": "EXPECTED_SELF_REFERENCE_FAIL_CLOSED",
            "directives": [{"slot_id": row["slot_id"], "media_type": row["media_type"],
                            "anchor_explicit": row["anchor_explicit"]} for row in before_slots],
            "terminal_video_error": "source slot slot-0 is not an earlier slot in this request",
            "image_artifacts": 0, "video_artifacts": 0, "native_wan_submits": 0,
            "public": before_public, "timing_ms": before_times,
            "native_pcm_joins": before_negative["native_pcm_joins"],
            "final_in_flight": before_negative["final_in_flight"],
        },
        "after": {
            "outcome": "REAL_FLUX_THEN_NATIVE_I2V_COMPLETE",
            "directives": [{"slot_id": row["slot_id"], "media_type": row["media_type"],
                            "anchor_slot_index": row.get("anchor_slot_index"),
                            "anchor_explicit": row.get("anchor_explicit", False)} for row in after_slots],
            "image": {key: image[key] for key in ("slot_id", "bytes", "sha256")},
            "image_decoded_dimensions": dimensions,
            "video": {key: video[key] for key in ("slot_id", "anchor_slot_index", "anchor_sha256", "bytes", "sha256")},
            "video_decoded": {"codec": stream["codec_name"], "pixel_format": stream["pix_fmt"],
                              "dimensions": [stream["width"], stream["height"]],
                              "frames": int(stream["nb_frames"]), "frame_rate": stream["r_frame_rate"]},
            "first_frame_mae": after_positive["first_frame_mae"],
            "motion_mae": after_positive["motion_mae"],
            "native_wan_source_exact": True,
            "native_wan_submits": 1,
            "native_wan_submit": {key: native_submit[key] for key in
                ("request_id", "packed_bytes", "packed_sha256", "image_bytes", "image_sha256",
                 "anchor_slot_index", "anchor_sha256", "anchor_crop", "anchor_explicit")},
            "public": after_public, "timing_ms": after_times,
            "native_pcm_joins": after_positive["pcm_joins"],
            "raw_verification": after_positive,
            "final_in_flight": after_positive["final_in_flight"],
        },
        "comparison": {
            "request_and_asr_identity_preserved": True,
            "terminal_identity_preserved": True,
            "real_flux_images": {"before": 0, "after": 1},
            "native_i2v_artifacts": {"before": 0, "after": 1},
            "native_wan_submits": {"before": 0, "after": 1},
            "first_text_and_voice_below_3000_ms_both_arms": first_response_pass,
            "stale_or_cross_session_events": 0,
            "cancel_and_immediate_exact_retry": True,
            "ns04_full_path_800ms_gate": "PASS" if after_times["full_barge_to_retry_first_audio"] < 800.0 else "NOT_MET",
            "promotion": promoted,
        },
        "boundaries": {
            "human_perceptual": "NOT_RUN",
            "physical_acoustic": "NOT_RUN",
            "full_production": "NOT_RUN",
            "resident_service": "NOT_CONTACTED",
            "training_optimizer_video_speed": "NOT_RUN",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--identities", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    try:
        result = verify(args.before, args.after, args.ffmpeg, args.ffprobe,
                        args.base_commit, args.execution_commit, args.identities)
        Path(args.output).write_bytes(json.dumps(result, indent=2, sort_keys=True,
            ensure_ascii=False, allow_nan=False).encode() + b"\n")
        print(json.dumps({"status": result["status"], "decision": result["decision"],
                          "output": args.output}, indent=2))
        return 0 if result["status"] == "PASS" else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"NS05 FIRST VIDEO VERIFY FAIL: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
