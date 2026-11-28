#!/usr/bin/env python3
"""Independent raw-wire/media verifier for the NS-05 voice co-creation bundle."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import wave


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_bytes())


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_ppm(payload):
    require(payload.startswith(b"P6\n"), "PPM header")
    header, raw = payload.split(b"\n255\n", 1)
    width, height = (int(value) for value in header.splitlines()[1].split())
    require(len(raw) == width * height * 3, "PPM bytes")
    return width, height, raw


def rgb(ffmpeg, path, selector=None):
    filters = []
    if selector:
        filters.append(selector)
    filters += ["scale=64:40:force_original_aspect_ratio=increase", "crop=64:40"]
    command = [ffmpeg, "-v", "error", "-i", str(path)]
    if filters:
        command += ["-vf", ",".join(filters)]
    command += ["-frames:v", "1", "-f", "image2pipe", "-vcodec", "ppm", "-"]
    result = subprocess.run(command, check=True, capture_output=True, timeout=60)
    return decode_ppm(result.stdout)[2]


def mae(left, right):
    require(len(left) == len(right) and bool(left), "RGB length")
    return sum(abs(a - b) for a, b in zip(left, right)) / len(left)


def verify_public(root, label, expected_terminal):
    record = load(root / f"{label}.json")
    require(record["label"] == label and record["terminal"]["kind"] == expected_terminal,
            f"{label} terminal")
    raw = (root / f"{label}.sse").read_bytes()
    require(sha_bytes(raw) == record["raw_sse_sha256"], f"{label} raw SSE hash")
    frames = load(root / f"{label}.frames.json")
    payloads = []
    for block in raw.split(b"\n\n"):
        for line in block.splitlines():
            if line.startswith(b"data:"):
                payloads.append(line[5:].lstrip().decode())
    require(payloads and payloads[-1] == "[DONE]", f"{label} DONE")
    expected_frames = [json.loads(payload) for payload in payloads if payload != "[DONE]"]
    require(expected_frames == frames, f"{label} frame join")
    require(all(frame.get("id") == record["request_id"] for frame in frames), f"{label} response id")
    require(all((frame.get("unified") or {}).get("session_id") in (None, "ns05-video-voice-coc-v1")
                for frame in frames), f"{label} cross-session")
    terminal_indices = [index for index, frame in enumerate(frames)
        if (frame.get("unified") or {}).get("kind") in ("complete", "cancelled", "error")]
    require(len(terminal_indices) == 1 and terminal_indices[0] == len(frames) - 1,
            f"{label} stale/post-terminal event")
    text = "".join(choice.get("delta", {}).get("content", "")
        for frame in frames for choice in frame.get("choices", [])
        if isinstance(choice.get("delta", {}).get("content"), str))
    require(text == record["text"] and sha_bytes(text.encode()) == record["text_sha256"],
            f"{label} clean text")
    groups = {}
    for frame in frames:
        unified = frame.get("unified") or {}
        if unified.get("kind") != "artifact":
            continue
        desc = unified["artifact"]
        groups.setdefault(desc["artifact_id"], []).append(desc)
    reconstructed = []
    for artifact_id, rows in groups.items():
        require([row["index"] for row in rows] == list(range(len(rows))), f"{label} chunk order")
        raw_media = b"".join(base64.b64decode(row["data"], validate=True) for row in rows)
        require(rows[-1]["bytes"] == len(raw_media) and rows[-1]["sha256"] == sha_bytes(raw_media),
                f"{label} media hash")
        matching = [row for row in record["artifacts"] if row["artifact_id"] == artifact_id]
        require(len(matching) == 1, f"{label} artifact record")
        item = matching[0]
        require((root / item["file"]).read_bytes() == raw_media and
                item["bytes"] == len(raw_media) and item["sha256"] == sha_bytes(raw_media),
                f"{label} artifact file")
        reconstructed.append(item)
    require(len(reconstructed) == len(record["artifacts"]), f"{label} artifact count")
    return record


def tap_ordinal(path):
    parts = path.name.split("-")
    return int(parts[2] if path.name.startswith("tts-output-") else parts[1])


def verify_pcm(root, record, submits, outputs):
    prefix = record["request_id"] + ":tts:"
    children = [message for message in submits if message.get("request_id", "").startswith(prefix)]
    require(children, f"no TTS submits: {record['label']}")
    public = [row for row in record["artifacts"] if row["media_type"] == "audio"]
    public_cursor = 0
    joins = []
    for submit in children:
        child = submit["request_id"]
        native = [message for message in outputs if message.get("type") == "stream"
                  and message.get("request_id") == child]
        terminals = [message for message in outputs if message.get("type") == "complete"
                     and message.get("request_id") == child]
        require(native and len(terminals) == 1 and terminals[0].get("success") is True,
                f"native TTS terminal: {child}")
        native_pcm = []
        for index, message in enumerate(native):
            chunk = message["chunk"]
            require(chunk["index"] == index and chunk["sample_rate"] == 24000
                    and chunk["channels"] == 1, f"native chunk metadata: {child}")
            require(public_cursor < len(public), f"public PCM missing: {child}")
            item = public[public_cursor]
            require(item["stream_index"] == index and
                    (root / item["file"]).read_bytes() == chunk["audio"],
                    f"public/native PCM mismatch: {child}:{index}")
            native_pcm.append(chunk["audio"])
            public_cursor += 1
        result = terminals[0]["result"]
        require(result.get("streamed") is True, f"native not streamed: {child}")
        with wave.open(io.BytesIO(result["data"]), "rb") as wav:
            require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 24000),
                    "native WAV format")
            terminal_pcm = wav.readframes(wav.getnframes())
        require(b"".join(native_pcm) == terminal_pcm, f"stream/terminal PCM: {child}")
        joins.append({"request_id": child, "chunk_count": len(native),
                      "pcm_bytes": len(terminal_pcm), "pcm_sha256": sha_bytes(terminal_pcm),
                      "terminal_wav_sha256": sha_bytes(result["data"]),
                      "exact_native_public_wav_join": True})
    require(public_cursor == len(public), f"unowned public PCM: {record['label']}")
    return joins


def verify(root, ffmpeg, ffprobe):
    root = Path(root).resolve()
    report = load(root / "report.json")
    prereg = load(root / "preregistration.json")
    requests = load(root / "requests.json")
    require(sha(root / "source.mp4") == prereg["source"]["derivative_sha256"], "source video")
    identity = load(root / "input-identity.json")
    for name, expected_sha in prereg["voice_cocreation_journey"]["input_audio_sha256"].items():
        require(sha(root / f"{name}.wav") == expected_sha
                and identity[name]["sha256"] == expected_sha, f"frozen voice: {name}")
    for name, body in requests.items():
        content = body["messages"][0]["content"]
        require([part["type"] for part in content] == ["video_url", "input_audio"],
                f"{name} voice/video-only input")
        require("frame_selection" not in json.dumps(body) and "[[" not in json.dumps(body)
                and body["narrative"]["visual_mode"] == "auto"
                and body["narrative"]["allow_text_only"] is True
                and body["narrative"]["continuity_mode"] == "i2v_chain",
                f"{name} zero-control default")
    expected = {"phone": "complete", "scene": "cancelled",
                "cyberpunk-initial": "cancelled", "cyberpunk-retry": "complete"}
    records = {label: verify_public(root, label, terminal) for label, terminal in expected.items()}
    require(len({row["request_id"] for row in records.values()}) == 4, "response identity")
    require(records["cyberpunk-initial"]["request_sha256"] ==
            records["cyberpunk-retry"]["request_sha256"], "immediate retry exact body")
    require(records["cyberpunk-initial"]["visual_slots"], "cancel was not at visual generation")
    initial_visual_legs = [leg for leg in records["cyberpunk-initial"]["terminal"]["legs"]
                           if leg.get("stage") in ("image", "video")]
    require(initial_visual_legs and all(leg.get("state") == "aborted"
            for leg in initial_visual_legs), "cancelled visual legs not settled")
    retry_visual_legs = [leg for leg in records["cyberpunk-retry"]["terminal"]["legs"]
                         if leg.get("stage") in ("image", "video")]
    require(len(retry_visual_legs) == 2 and all(leg.get("state") == "completed"
            for leg in retry_visual_legs), "retry visual terminal legs")
    for label, record in records.items():
        require(record["asr_transcripts"], f"ASR missing: {label}")
    images = [row for row in records["cyberpunk-retry"]["artifacts"] if row["media_type"] == "image"]
    videos = [row for row in records["cyberpunk-retry"]["artifacts"] if row["media_type"] == "video"]
    require(len(images) == 1 and len(videos) == 1, "FLUX/I2V count")
    image, video = images[0], videos[0]
    image_bytes = (root / image["file"]).read_bytes()
    require(video["anchor_sha256"] == sha_bytes(image_bytes), "public anchor")
    import msgpack
    submits, outputs, wan_submits = [], [], []
    for path in sorted(root.glob("*.msgpack"), key=tap_ordinal):
        message = msgpack.unpackb(path.read_bytes(), raw=False)
        if path.name.startswith("tts-output-") and message.get("type") in ("stream", "complete"):
            outputs.append(message)
        elif path.name.startswith("tts-") and message.get("type") == "submit":
            submits.append(message)
        elif path.name.startswith("video-") and message.get("type") == "submit":
            wan_submits.append(message)
    joins = {}
    for label in ("phone", "cyberpunk-retry"):
        computed = verify_pcm(root, records[label], submits, outputs)
        reported = report["native_pcm_joins"][label]
        require(len(computed) == len(reported), f"PCM join count: {label}")
        for actual, expected_row in zip(computed, reported):
            require(actual["request_id"] == expected_row["request_id"] and
                    actual["chunk_count"] == expected_row["chunk_count"] and
                    actual["pcm_bytes"] == expected_row["pcm_bytes"] and
                    actual["pcm_sha256"] == expected_row["pcm_sha256"] and
                    actual["terminal_wav_sha256"] == expected_row["terminal_wav_sha256"],
                    f"reported PCM join: {label}")
        joins[label] = computed
    retry_prefix = records["cyberpunk-retry"]["request_id"] + ":slot:"
    selected_wan = [message for message in wan_submits
                    if message.get("request_id", "").startswith(retry_prefix)]
    require(len(selected_wan) == 1, "native Wan submit count")
    inputs = selected_wan[0]["data"]["request"]["inputs"]
    require(inputs["image"] == image_bytes and inputs["anchor_sha256"] == sha_bytes(image_bytes)
            and inputs["anchor_crop"] == "center_cover"
            and isinstance(inputs.get("anchor_explicit", False), bool),
            "native Wan exact source")
    log = (root / "video.log").read_text(errors="replace")
    require("backend=native-cuda-wan" in log and "i2v=1" in log, "native Wan log")
    probe = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,nb_frames,r_frame_rate,pix_fmt",
        "-of", "json", str(root / video["file"])], check=True, capture_output=True,
        text=True, timeout=30)
    stream = json.loads(probe.stdout)["streams"][0]
    require(stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
            and (stream["width"], stream["height"]) == (704, 480)
            and int(stream["nb_frames"]) == 29 and stream["r_frame_rate"] == "16/1", "video geometry")
    source_eval = rgb(ffmpeg, root / image["file"])
    first_eval = rgb(ffmpeg, root / video["file"], "select=eq(n\\,0)")
    last_eval = rgb(ffmpeg, root / video["file"], "select=eq(n\\,28)")
    first_mae = mae(source_eval, first_eval)
    motion_mae = mae(first_eval, last_eval)
    require(first_mae <= 48.0 and motion_mae >= 1.0, "continuity/motion gate")
    require(abs(first_mae - report["first_frame_mae"]) < 1e-9 and
            abs(motion_mae - report["motion_mae"]) < 1e-9, "reported continuity")
    require("[[" not in records["cyberpunk-retry"]["text"] and
            "]]" not in records["cyberpunk-retry"]["text"] and
            records["cyberpunk-retry"]["text"].strip(), "clean narration")
    barge = load(root / "barge.json")
    require(barge["interrupted_request_id"] == records["scene"]["request_id"]
            and barge["initial_request_id"] == records["cyberpunk-initial"]["request_id"]
            and barge["barge_start_monotonic_ns"] == report["barge_start_monotonic_ns"],
            "raw barge identity")
    barge_ms = (records["cyberpunk-retry"]["first_audio_monotonic_ns"] -
                barge["barge_start_monotonic_ns"]) / 1e6
    require(abs(barge_ms - report["full_barge_to_retry_first_audio_ms"]) < 1e-6
            and barge_ms > 0, "barge clocks")
    cleanup = load(root / "cleanup.json")
    require(cleanup["post"]["memory_used_mib"] <= 16 and cleanup["in_flight"]["in_flight"] == 0,
            "cleanup/in_flight")
    require(report["status"] == "PASS" and report["response_identity_unique"]
            and report["no_stale_or_cross_session_events"], "report lifecycle")
    return {"status": "PASS", "response_ids": {label: row["request_id"] for label, row in records.items()},
        "asr_transcripts": {label: row["asr_transcripts"] for label, row in records.items()},
        "image_sha256": image["sha256"], "video_sha256": video["sha256"],
        "pcm_joins": joins, "first_frame_mae": first_mae, "motion_mae": motion_mae,
        "native_wan_source_exact": True, "cancel_retry_exact": True,
        "stale_or_cross_session_events": 0, "final_in_flight": 0,
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
        require(sha(path) == expected[relative], f"manifest: {relative}")
    return {"status": "PASS", "files": len(actual),
            "bytes": sum(path.stat().st_size for path in actual.values()),
            "sha256sums_sha256": sha(root / "SHA256SUMS")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    try:
        result = (verify_manifest(args.root) if args.manifest_only else
                  verify(args.root, args.ffmpeg, args.ffprobe))
        Path(args.output).write_bytes(json.dumps(result, indent=2, sort_keys=True,
            ensure_ascii=False).encode() + b"\n")
        print(json.dumps({"status": "PASS", "output": args.output}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"NS05 COCREATION VERIFY FAIL: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
