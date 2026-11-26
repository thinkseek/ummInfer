#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent fresh-process verifier for flux2_i2v_continuity_live evidence.

Uses only the standard library plus FFmpeg/ffprobe executables. It deliberately
imports neither the live driver nor the narrative fixture/implementation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical_sha256(value):
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def decode_ppm(raw):
    require(raw.startswith(b"P6\n"), "FFmpeg output is not binary PPM")
    header, pixels = raw.split(b"\n255\n", 1)
    dimensions = header.splitlines()[1].split()
    require(len(dimensions) == 2, "PPM dimensions missing")
    width, height = map(int, dimensions)
    require(len(pixels) == width * height * 3, "PPM RGB size mismatch")
    return pixels


def decode(media, *, frame=None, evaluation=False, ffmpeg="ffmpeg"):
    command = [ffmpeg, "-v", "error", "-i", str(media)]
    filters = []
    if frame is not None:
        filters.append(f"select=eq(n\\,{frame})")
    if evaluation:
        filters += ["scale=64:40:force_original_aspect_ratio=increase", "crop=64:40"]
    if filters:
        command += ["-vf", ",".join(filters)]
    command += ["-frames:v", "1", "-f", "image2pipe", "-vcodec", "ppm", "-"]
    return decode_ppm(subprocess.run(
        command, check=True, capture_output=True, timeout=60).stdout)


def mae(first, second):
    require(len(first) == len(second) and len(first) > 0 and len(first) % 3 == 0,
            "MAE inputs must be equal non-empty RGB buffers")
    return sum(abs(a - b) for a, b in zip(first, second)) / len(first)


def verify(root, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    root = Path(root)
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    require(report["format"] == "umminfer-flux2-wan22-i2v-continuity-v1",
            "unknown report format")
    require(report["status"] == "GENERATED_AWAITING_INDEPENDENT_VERIFICATION",
            "report has an invalid pre-verification state")
    require(report["physical_gpu"] == 1, "wrong physical GPU")
    require(len(report["base_commit"]) == 40 and len(report["execution_commit"]) == 40
            and report["provenance"]["base_commit"] == report["base_commit"]
            and report["provenance"]["execution_commit"] == report["execution_commit"],
            "base/execution commit binding missing")
    require(report["public_vlm_narrative_orchestration"] == "NOT_RUN"
            and report["perceptual_quality"] == "NOT_RUN", "scope markers changed")

    manifest_lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    manifest = {}
    for line in manifest_lines:
        digest, name = line.split("  ", 1)
        require(len(digest) == 64 and Path(name).name == name and name not in manifest,
                f"invalid manifest row: {line}")
        manifest[name] = digest
    expected = {path.name for path in root.iterdir() if path.is_file()
                and path.name not in {"SHA256SUMS", "independent-verification.json",
                                      "summary.json"}}
    require(set(manifest) == expected, "manifest does not cover exactly the retained input files")
    for name, digest in manifest.items():
        require(sha256((root / name).read_bytes()) == digest,
                f"manifest SHA-256 mismatch: {name}")
    for model in report["models"].values():
        require(model["file_count"] == len(model["files"])
                and model["tree_sha256"] == canonical_sha256(model["files"]),
                f"model tree identity mismatch: {model['model_id']}")

    source = report["source"]
    source_bytes = (root / "source.png").read_bytes()
    source_sha = sha256(source_bytes)
    require(source["success"] and source["result"]["backend"] == "diffusers"
            and source["result"]["model"] == "black-forest-labs/FLUX.2-klein-4B"
            and source["result"]["seed"] == 42 and source["result"]["steps"] == 4,
            "source is not the required real FLUX execution")
    require(source_bytes == (root / source["media_file"]).read_bytes()
            and source_sha == source["media_sha256"], "source bytes/hash differ")
    source_full = decode(root / "source.png", ffmpeg=ffmpeg)
    source_eval = decode(root / "source.png", evaluation=True, ffmpeg=ffmpeg)
    require(len(source_full) == 512 * 512 * 3 and len(source_eval) == 64 * 40 * 3,
            "source decoded dimensions mismatch")
    require(source_full == (root / "source.rgb512x512").read_bytes()
            and source_eval == (root / "source.rgb64x40").read_bytes(),
            "retained source RGB differs from fresh decode")

    runs = report["measured_runs"]
    require([run["ordinal"] for run in runs] == list(range(6)), "run order changed")
    require([run["mode"] for run in runs] == ["i2v", "t2v"] * 3,
            "pre-registered I2V/T2V order changed")
    i2v, t2v = [], []
    expected_common = {
        "prompt": "The red lantern sways gently while the camera slowly orbits above the misty lake",
        "size": "704x480", "num_frames": 29, "steps": 20,
        "guidance_scale": 5.0, "seed": 42, "output_format": "mp4",
    }
    anchor_fields = {"image", "anchor_explicit", "anchor_slot_index",
                     "anchor_crop", "anchor_sha256"}
    native_log = (root / "video.stage.log").read_text(encoding="utf-8").splitlines()
    generation_lines = [line for line in native_log if "video request backend=" in line]
    for run in runs:
        require(run["success"], f"measured run failed: {run['request_id']}")
        require({key: value for key, value in run["inputs"].items()
                 if key not in anchor_fields} == expected_common,
                f"non-anchor settings differ: {run['request_id']}")
        result = run["result"]
        require(result["backend"] == "native-cuda-wan"
                and result["model"] == "Wan2.2-TI2V-5B-Diffusers"
                and result["seed"] == 42 and result["steps"] == 20
                and result["frames"] == 29 and result["size"] == "704x480",
                f"native Wan identity/settings mismatch: {run['request_id']}")
        matching_logs = [line for line in generation_lines
                         if f"request_id={run['request_id']}" in line]
        require(len(matching_logs) == 1
                and "backend=native-cuda-wan" in matching_logs[0]
                and run["native_path"] in matching_logs[0]
                and run["native_log_sha256"] == sha256(matching_logs[0].encode()),
                f"native execution log mismatch: {run['request_id']}")
        media = root / run["media_file"]
        require(sha256(media.read_bytes()) == run["media_sha256"]
                and media.stat().st_size == run["media_bytes"], "media hash/bytes mismatch")
        probe = json.loads(subprocess.run([
            ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=codec_name,width,height,nb_frames,r_frame_rate,pix_fmt",
            "-of", "json", str(media)], check=True, capture_output=True,
            text=True, timeout=30).stdout)["streams"][0]
        require(probe == run["video_stream"] and probe["codec_name"] == "h264"
                and probe["pix_fmt"] == "yuv420p"
                and (probe["width"], probe["height"]) == (704, 480)
                and probe["nb_frames"] == "29" and probe["r_frame_rate"] == "16/1",
                f"fresh ffprobe mismatch: {run['request_id']}")
        first = decode(media, frame=0, evaluation=True, ffmpeg=ffmpeg)
        last = decode(media, frame=28, evaluation=True, ffmpeg=ffmpeg)
        first_full = decode(media, frame=0, ffmpeg=ffmpeg)
        last_full = decode(media, frame=28, ffmpeg=ffmpeg)
        require(len(first_full) == 704 * 480 * 3 and len(last_full) == len(first_full),
                "full video RGB dimensions mismatch")
        expected_rgb = {
            "first_eval": first, "last_eval": last,
            "first_full": first_full, "last_full": last_full,
        }
        for label, raw in expected_rgb.items():
            descriptor = run["decoded_rgb"][label]
            require(raw == (root / descriptor["file"]).read_bytes()
                    and sha256(raw) == descriptor["sha256"] and len(raw) == descriptor["bytes"],
                    f"retained {label} RGB differs: {run['request_id']}")
        first_mae = round(mae(source_eval, first), 6)
        motion_mae = round(mae(first, last), 6)
        require(first_mae == run["first_frame_mae"] and motion_mae == run["motion_mae"],
                f"numeric proxy mismatch: {run['request_id']}")
        require(run["metric_version"] == "center-cover-64x40-rgb8-frame0-frame28-v1",
                "metric version changed")
        if run["mode"] == "i2v":
            require(run["source_sha256"] == source_sha
                    and run["inputs"]["image"]["sha256"] == source_sha
                    and run["inputs"]["anchor_sha256"] == source_sha
                    and run["inputs"]["anchor_explicit"] is True
                    and run["inputs"]["anchor_slot_index"] == 0
                    and run["native_path"] == "i2v=1", "I2V exact binding missing")
            require(first_mae <= 48.0 and motion_mae >= 1.0,
                    "frozen I2V numeric gate failed")
            i2v.append(run)
        else:
            require(anchor_fields.isdisjoint(run["inputs"])
                    and run["native_path"] == "i2v=0", "T2V control is anchored")
            t2v.append(run)
    require(len({run["media_sha256"] for run in i2v}) == 1
            and len({run["media_sha256"] for run in t2v}) == 1
            and i2v[0]["media_sha256"] != t2v[0]["media_sha256"],
            "fixed-seed reproducibility or I2V/T2V path distinction failed")
    negatives = report["negative_controls"]
    require([item["control"] for item in negatives] == ["wrong_hash", "tampered_source"],
            "negative controls changed")
    for item in negatives:
        require(not item["success"] and item["error"] == "video.image does not match anchor_sha256"
                and "media_file" not in item and "media_sha256" not in item
                and not any(f"request_id={item['request_id']}" in line
                            for line in generation_lines),
                "negative control generated media or failed for the wrong reason")
    require(negatives[0]["inputs"]["image"]["sha256"] == source_sha
            and negatives[0]["inputs"]["anchor_sha256"] == "0" * 64,
            "wrong-hash control does not contain the exact source")
    tampered = (root / "tampered-source.png").read_bytes()
    require(sha256(tampered) != source_sha
            and negatives[1]["inputs"]["image"]["sha256"] == sha256(tampered)
            and negatives[1]["inputs"]["anchor_sha256"] == source_sha,
            "tampered-source control does not bind changed bytes to the stale hash")

    retained_records = [json.loads(line) for line in
                        (root / "records.jsonl").read_text(encoding="utf-8").splitlines()
                        if line]
    require(retained_records == [source, *negatives, *runs],
            "records.jsonl differs from final enriched report records")

    for item in [source, *negatives, *runs]:
        observed_latency = (item["complete_monotonic_ns"]
                            - item["submit_monotonic_ns"]) / 1e6
        require(observed_latency >= 0 and item["latency_ms"] == observed_latency,
                "nonmonotonic or inconsistent request timing")
    stages = report["stages"]
    require(len(stages) == 2 and [stage["stage"] for stage in stages] == ["image", "video"]
            and all(stage["returncode"] == 0 and stage["stop_error"] is None
                    for stage in stages), "stage cleanup failed")
    for stage, items in zip(stages, ([source], [*negatives, *runs])):
        previous_complete = stage["ready_monotonic_ns"]
        require(stage["start_monotonic_ns"] <= previous_complete, "ready preceded stage start")
        for item in items:
            require(previous_complete <= item["submit_monotonic_ns"]
                    <= item["complete_monotonic_ns"] <= stage["stop_monotonic_ns"],
                    "request outside stage readiness/serial/stop interval")
            previous_complete = item["complete_monotonic_ns"]
    require(stages[0]["stop_monotonic_ns"] <= stages[1]["start_monotonic_ns"],
            "native model stages overlapped")
    require(all(report["gpu1"][key]["index"] == 1
                and report["gpu1"][key]["memory_used_mib"] <= 16
                for key in ("before", "between", "after")), "GPU 1 cleanup evidence failed")

    medians = {
        "source_request": round(statistics.median([source["latency_ms"]]), 6),
        "i2v_request": round(statistics.median(run["latency_ms"] for run in i2v), 6),
        "t2v_request": round(statistics.median(run["latency_ms"] for run in t2v), 6),
        "i2v_denoise": round(statistics.median(run["result"]["denoise_ms"] for run in i2v), 6),
        "t2v_denoise": round(statistics.median(run["result"]["denoise_ms"] for run in t2v), 6),
    }
    require(medians == report["latency_median_ms"], "latency medians differ")
    counts = {
        "source_images": 1, "i2v_videos": len(i2v), "t2v_control_videos": len(t2v),
        "successful_requests": 1 + len(runs), "rejected_anchor_requests": len(negatives),
    }
    require(counts == report["real_generation_counts"], "real generation counts differ")
    return {
        "status": "CONTROLLED_LIVE", "verified_manifest_files": len(manifest),
        "source_sha256": source_sha,
        "i2v_sha256": i2v[0]["media_sha256"], "t2v_sha256": t2v[0]["media_sha256"],
        "i2v_first_frame_mae": [run["first_frame_mae"] for run in i2v],
        "i2v_motion_mae": [run["motion_mae"] for run in i2v],
        "t2v_first_frame_mae": [run["first_frame_mae"] for run in t2v],
        "t2v_motion_mae": [run["motion_mae"] for run in t2v],
        "latency_median_ms": medians,
        "negative_controls_rejected": len(negatives), "real_generation_counts": counts,
        "perceptual_quality": "NOT_RUN", "public_vlm_narrative_orchestration": "NOT_RUN",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    require(not args.out.exists(), f"verification output already exists: {args.out}")
    result = verify(args.evidence, args.ffmpeg, args.ffprobe)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("INDEPENDENT_FRESH_PROCESS_VERIFICATION: PASS")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
