#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent fresh-process verifier for public_narrative_i2v_live evidence.

This imports neither the live driver nor trajectory/replay producers. It parses
raw SSE and retained MessagePack submits, rebuilds every artifact from chunks,
rechecks source/anchor/TTS binding, decodes RGB, and optionally rehashes the
exact external binaries and model trees.
"""
import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import wave


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def parse_frames(raw):
    frames = []
    cursor = 0
    while cursor < len(raw):
        end = raw.find(b"\n\n", cursor)
        require(end >= 0, "truncated raw SSE frame")
        payload = raw[cursor:end + 2]
        data = []
        for line in payload[:-2].split(b"\n"):
            if line.startswith(b"data:"):
                value = line[5:]
                data.append(value[1:] if value.startswith(b" ") else value)
        require(data, "non-data SSE frame is not part of this capture")
        frames.append((cursor, payload, b"\n".join(data)))
        cursor = end + 2
    return frames


def decode_ppm(payload):
    require(payload.startswith(b"P6\n"), "expected P6 PPM")
    header, raw = payload.split(b"\n255\n", 1)
    width, height = (int(value) for value in header.splitlines()[1].split())
    require(len(raw) == width * height * 3, "RGB byte count mismatch")
    return width, height, raw


def rgb(ffmpeg, path, selector=None, evaluation=False):
    filters = []
    if selector:
        filters.append(selector)
    if evaluation:
        filters += ["scale=64:40:force_original_aspect_ratio=increase", "crop=64:40"]
    command = [ffmpeg, "-v", "error", "-i", str(path)]
    if filters:
        command += ["-vf", ",".join(filters)]
    command += ["-frames:v", "1", "-f", "image2pipe", "-vcodec", "ppm", "-"]
    result = subprocess.run(command, check=True, capture_output=True, timeout=60)
    return decode_ppm(result.stdout)


def mae(first, second):
    require(len(first) == len(second) and bool(first), "invalid RGB comparison")
    return sum(abs(a - b) for a, b in zip(first, second)) / len(first)


def spoken_projection(text):
    text = re.sub(r"(?m)^ {0,3}(?:#+[ \t]+|>[ \t]?|[*+-][ \t]+)", "", text)
    text = re.sub(r"(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", "", text)
    text = text.replace("**", "").replace("__", "").replace("~~", "")
    text = text.replace("`", "")
    return "".join(text.split())


def has_native_presentation_tokens(text):
    return bool(re.search(r"(?:\*\*|__|~~|`)", text) or
                re.search(r"(?m)^ {0,3}#+[ \t]+", text) or
                re.search(r"(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", text))


def verify_model(identity):
    root = Path(identity["root"])
    files = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        files[path.relative_to(root).as_posix()] = file_sha(path)
    require(files == identity["files"], f"model file identity changed under {root}")
    require(sha(canonical(files)) == identity["tree_sha256"],
            f"model tree digest changed under {root}")
    return {"file_count": len(files), "tree_sha256": identity["tree_sha256"],
            "identity": "REHASHED_EXACT"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--skip-provenance-rehash", action="store_true")
    args = parser.parse_args()
    root = args.evidence.resolve()
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    require(report["format"] == "umminfer-public-narrative-i2v-v1", "unknown report format")
    require(report["result"] == "CONTROLLED_LIVE_PASS", "live gate did not pass")
    physical_gpu = report["physical_gpu"]
    require(physical_gpu in (0, 1, 7), "wrong physical GPU")
    scenario = report.get("scenario", "lantern")
    request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    require(sha(canonical(request)) == report["request_sha256"], "request digest mismatch")
    if scenario == "ns01-black-hole":
        require(request["messages"] == [{"role": "user", "content":
                "Use one labeled still and one short animation to explain black-hole "
                "accretion disks and gravitational lensing simply, in plain prose without "
                "headings or bullets."}]
                and "[[" not in request["messages"][0]["content"]
                and request["narrative"]["visual_mode"] == "auto"
                and request["narrative"]["continuity_mode"] == "i2v_chain",
                "NS-01 did not use the frozen natural autonomous prompt")
    elif scenario == "ns01-chinese-markdown-long":
        require(request["messages"][0]["content"].startswith(
                "用通俗方式讲清黑洞吸积盘和引力透镜，配示意图和动态演示。")
                and "三个内容充实的中文长段落" in request["messages"][0]["content"]
                and "Markdown" in request["messages"][0]["content"]
                and all(term in request["messages"][0]["content"] for term in
                        ("accretion disk", "gravitational lensing", "event horizon",
                         "Eddington limit"))
                and request["max_tokens"] == 768
                and request["narrative"]["visual_mode"] == "auto"
                and request["narrative"]["continuity_mode"] == "i2v_chain"
                and "teacache_thresh" not in request["video"],
                "Chinese NS-01 did not use the frozen long autonomous prompt")
    else:
        require(scenario == "lantern", "unknown narrative scenario")
    records = [json.loads(line) for line in
               (root / report["trajectory"]).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    require(len(records) == 1, "expected one public trajectory")
    record = records[0]
    replay = record["sse_replay"]
    raw_path = root / "capture" / "artifacts" / replay["raw_ref"]
    raw = raw_path.read_bytes()
    require(sha(raw) == replay["raw_sha256"] == report["raw_sse"]["raw_sha256"],
            "raw SSE digest mismatch")
    require(len(raw) == replay["raw_bytes"], "raw SSE byte count mismatch")
    frames = parse_frames(raw)
    require(len(frames) == len(replay["frames"]), "frame count mismatch")
    previous_tick = -1
    events = []
    done = 0
    for index, (offset, frame_raw, payload) in enumerate(frames):
        desc = replay["frames"][index]
        require(desc["index"] == index and desc["offset"] == offset
                and desc["bytes"] == len(frame_raw) and desc["sha256"] == sha(frame_raw),
                f"raw frame binding mismatch at {index}")
        require(desc["tick"] >= previous_tick, "nonmonotonic raw frame timing")
        previous_tick = desc["tick"]
        if payload == b"[DONE]":
            done += 1
            require(index == len(frames) - 1 and desc["kind"] == "done",
                    "[DONE] is missing or not final")
            continue
        require(desc["kind"] == "event" and desc["data_sha256"] == sha(payload),
                f"event frame descriptor mismatch at {index}")
        frame = json.loads(payload)
        if frame.get("unified") is None:
            require(frame.get("object") == "chat.completion.chunk"
                    and isinstance(frame.get("choices"), list),
                    "non-unified frame is not an OpenAI transport chunk")
            continue
        events.append((index, frame["unified"]))
    require(done == 1 and len(events) == len(record["events"]),
            "public event/[DONE] count mismatch")
    require([event.get("seq") for _, event in events] == list(range(len(events))),
            "public unified sequence has a gap")
    source_ticks = [event["source_event_tick"] for event in record["events"]]
    require(source_ticks == sorted(source_ticks), "normalized event ticks are nonmonotonic")
    for event, (frame_index, raw_event) in zip(record["events"], events):
        require(event["source_event_index"] == frame_index
                and event["source_event_sha256"] == replay["frames"][frame_index]["sha256"],
                "normalized event lacks exact raw-frame binding")
    terminals = [event for _, event in events
                 if event["kind"] in ("complete", "cancelled", "error")]
    require(len(terminals) == 1 and events[-1][1] is terminals[0]
            and terminals[0]["kind"] == "complete", "terminal contract failed")
    clean = "".join(event.get("text", "") for _, event in events
                    if event["kind"] == "vlm_delta")
    require(clean == record["clean_text"] == report["clean_text"],
            "clean prose reconstruction differs")
    require("[[" not in clean and "]]" not in clean and "source=" not in clean
            and "[[image:" not in clean and "[[video:" not in clean,
            "directive/source suffix leaked into prose")
    slots = [event for _, event in events if event["kind"] == "visual_slot"]
    require(len(slots) == 2, "expected exactly one image and one video slot")
    source_slot, video_slot = slots
    require(source_slot["slot_id"] == "slot-0" and source_slot["stage"] == "image"
            and source_slot["slot_index"] == 0, "source slot identity mismatch")
    require(video_slot["slot_id"] == "slot-1" and video_slot["stage"] == "video"
            and video_slot["slot_index"] == 1
            and video_slot.get("continuity_mode") == "i2v_chain"
            and video_slot.get("anchor_slot_index") == 0
            and video_slot.get("anchor_explicit") is True
            and "anchor_sha256" not in video_slot,
            "explicit video slot metadata mismatch")
    require("source=" not in video_slot["prompt"] and "[[" not in video_slot["prompt"],
            "source suffix leaked into video render prompt")
    first_ticks = {"ttft_ns": None, "ttfa_ns": None, "first_image_ns": None,
                   "video_first_public_chunk_ns": None, "video_public_complete_ns": None}
    for event, (_frame_index, raw_event) in zip(record["events"], events):
        tick = event["source_event_tick"]
        if raw_event["kind"] == "vlm_delta" and raw_event.get("text") \
                and first_ticks["ttft_ns"] is None:
            first_ticks["ttft_ns"] = tick
        if raw_event["kind"] != "artifact":
            continue
        desc = raw_event["artifact"]
        if desc.get("media_type") == "audio" and first_ticks["ttfa_ns"] is None:
            first_ticks["ttfa_ns"] = tick
        if desc.get("media_type") == "video":
            if first_ticks["video_first_public_chunk_ns"] is None:
                first_ticks["video_first_public_chunk_ns"] = tick
            if desc["index"] + 1 == desc["total"]:
                first_ticks["video_public_complete_ns"] = tick
        if desc.get("media_type") == "image" and desc.get("slot_id") == "slot-0" \
                and desc["index"] + 1 == desc["total"]:
            first_ticks["first_image_ns"] = tick
    expected_timings = {key: (None if tick is None else tick / 1e6)
                        for key, tick in first_ticks.items()}
    expected_timings.update(unit="ms_from_capture_start", clock="python.time.perf_counter_ns")
    if "interactive_timings" in report:
        require(report["interactive_timings"] == expected_timings,
                "interactive arrival timings differ from raw SSE ticks")
        require(all(first_ticks[key] is not None for key in first_ticks),
                "NS-01 is missing text/audio/image/video arrival evidence")
    else:
        require(scenario == "lantern", "new NS-01 evidence lacks arrival timings")
    grouped = {}
    for frame_index, event in events:
        if event["kind"] != "artifact":
            continue
        desc = event["artifact"]
        group = grouped.setdefault(desc["artifact_id"], [])
        group.append((frame_index, event["seq"], desc))
    rebuilt = {}
    chunk_total = 0
    for artifact_id, chunks in grouped.items():
        chunk_total += len(chunks)
        chunks.sort(key=lambda item: item[2]["index"])
        total = chunks[0][2]["total"]
        require([item[2]["index"] for item in chunks] == list(range(total)),
                f"artifact chunk gap for {artifact_id}")
        payload = b"".join(base64.b64decode(item[2]["data"], validate=True)
                           for item in chunks)
        final = chunks[-1][2]
        require(final["bytes"] == len(payload) and final["sha256"] == sha(payload),
                f"artifact digest/bytes mismatch for {artifact_id}")
        rebuilt[artifact_id] = (payload, final, chunks)
    by_media = {}
    for payload, final, chunks in rebuilt.values():
        by_media.setdefault(final["media_type"], []).append((payload, final, chunks))
    require(set(by_media) == {"image", "video", "audio"}, "missing generated modality")
    require(len(by_media["image"]) == len(by_media["video"]) == 1,
            "expected exactly one source image and one video")
    source_raw, source_desc, source_chunks = by_media["image"][0]
    video_raw, video_desc, video_chunks = by_media["video"][0]
    require(source_desc["slot_id"] == "slot-0" and video_desc["slot_id"] == "slot-1",
            "artifact slot attribution mismatch")
    require(video_chunks[0][1] > source_chunks[-1][1],
            "video artifact predates final source artifact chunk")
    require(video_desc.get("anchor_slot_index") == 0
            and video_desc.get("anchor_explicit") is True
            and video_desc.get("anchor_sha256") == sha(source_raw),
            "public video artifact does not bind exact source bytes")
    terminal_legs = terminals[0]["legs"]
    leg_by_slot = {leg.get("slot_id"): leg for leg in terminal_legs if leg.get("slot_id")}
    require(leg_by_slot["slot-0"]["state"] == "completed"
            and leg_by_slot["slot-1"]["state"] == "completed"
            and leg_by_slot["slot-1"].get("anchor_sha256") == sha(source_raw),
            "terminal visual legs/source binding mismatch")
    require(any(leg.get("stage") == "tts" and leg.get("state") == "completed"
                for leg in terminal_legs), "TTS terminal leg did not complete")
    artifact_records = record["artifacts"]
    require(len(artifact_records) == len(rebuilt) == report["public_artifact_count"],
            "CAS artifact count mismatch")
    for item in artifact_records:
        cas = root / "capture" / "artifacts" / item["ref"]
        payload = cas.read_bytes()
        require(sha(payload) == item["sha256"] and len(payload) == item["bytes"],
                f"CAS identity mismatch for {item['ref']}")
        match = next((entry for entry in rebuilt.values() if entry[0] == payload), None)
        require(match is not None, f"CAS payload absent from raw chunks: {item['ref']}")
    require(chunk_total == report["public_artifact_chunk_count"],
            "public artifact chunk count mismatch")

    reconstructed = root / "reconstructed"
    require(not reconstructed.exists(), "reconstruction directory already exists")
    reconstructed.mkdir()
    (reconstructed / "source.png").write_bytes(source_raw)
    (reconstructed / "video.mp4").write_bytes(video_raw)
    (reconstructed / "raw.sse").write_bytes(raw)
    (reconstructed / "events.json").write_text(json.dumps(
        [event for _, event in events], indent=2, sort_keys=True) + "\n")
    audio = []
    for index, (pcm, desc, _chunks) in enumerate(by_media["audio"]):
        require(desc["mime_type"] == "audio/pcm" and desc.get("format") == "pcm16"
                and desc.get("sample_rate") == 24000 and desc.get("channels") == 1
                and len(pcm) % 2 == 0 and pcm, "invalid PCM16 public artifact")
        samples = [int.from_bytes(pcm[pos:pos + 2], "little", signed=True)
                   for pos in range(0, len(pcm), 2)]
        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
        require(rms > 1.0, "silent PCM artifact")
        name = f"audio-{index}.pcm"
        (reconstructed / name).write_bytes(pcm)
        audio.append({"file": name, "stream_index": desc.get("stream_index"),
                      "sha256": sha(pcm), "bytes": len(pcm), "rms": rms})
    require(len(audio) == len(report["audio"]), "speech artifact count mismatch")

    import msgpack
    video_submit = report["video_submit"]
    packed = (root / video_submit["packed_file"]).read_bytes()
    require(sha(packed) == video_submit["packed_sha256"], "Wan packed submit hash mismatch")
    message = msgpack.unpackb(packed, raw=False)
    inputs = message["data"]["request"]["inputs"]
    require(inputs["image"] == source_raw, "Wan did not receive immutable source bytes")
    require(sha(inputs["image"]) == inputs["anchor_sha256"] == sha(source_raw)
            and inputs["anchor_explicit"] is True and inputs["anchor_slot_index"] == 0
            and inputs["anchor_crop"] == "center_cover",
            "native Wan submit anchor binding mismatch")
    require("source=" not in inputs["prompt"], "source suffix reached native Wan prompt")
    (reconstructed / "video-submit.msgpack").write_bytes(packed)
    tts_text_parts = []
    for submit in report["tts_submits"]:
        packed_tts = (root / submit["packed_file"]).read_bytes()
        require(sha(packed_tts) == submit["packed_sha256"], "TTS submit hash mismatch")
        unpacked = msgpack.unpackb(packed_tts, raw=False)
        text = unpacked["data"]["request"]["inputs"]["text"]
        require(text == submit["text"] and sha(text.encode()) == submit["text_sha256"],
                "TTS text submit identity mismatch")
        tts_text_parts.append(text)
    tts_text = "".join(tts_text_parts)
    require(spoken_projection(tts_text) == spoken_projection(clean)
            and "[[" not in tts_text and "source=" not in tts_text,
            "TTS input differs beyond trimmed span whitespace or leaks a directive")
    require(not any(has_native_presentation_tokens(text) for text in tts_text_parts),
            "projection comparison hid Markdown tokens in actual native TTS input")

    public_cursor = 0
    verified_tts_legs = []
    for leg_index, submit in enumerate(report["tts_submits"]):
        child_id = submit["request_id"]
        native_chunks = [item for item in report["tts_output"]
                         if item.get("message_type") == "stream"
                         and item.get("request_id") == child_id]
        terminals_tts = [item for item in report["tts_output"]
                         if item.get("message_type") == "complete"
                         and item.get("request_id") == child_id]
        require(native_chunks and len(terminals_tts) == 1
                and terminals_tts[0].get("success") is True
                and terminals_tts[0].get("streamed") is True,
                f"incomplete native TTS leg: {child_id}")
        native_pcm_parts = []
        for expected_index, native in enumerate(native_chunks):
            native_packed = (root / native["packed_file"]).read_bytes()
            require(sha(native_packed) == native["packed_sha256"],
                    "native TTS output packed hash mismatch")
            native_message = msgpack.unpackb(native_packed, raw=False)
            require(native_message["request_id"] == child_id
                    and native_message["chunk"]["index"] == expected_index,
                    f"native TTS chunk cursor mismatch: {child_id}")
            native_audio = native_message["chunk"]["audio"]
            require(sha(native_audio) == native["audio_sha256"],
                    "native TTS chunk hash mismatch")
            require(public_cursor < len(by_media["audio"]),
                    f"public PCM missing for {child_id}:{expected_index}")
            public_pcm, public_desc, _ = by_media["audio"][public_cursor]
            require(public_desc.get("stream_index") == expected_index
                    and public_pcm == native_audio,
                    f"public PCM does not belong to {child_id}:{expected_index}")
            native_pcm_parts.append(native_audio)
            public_cursor += 1
        complete_packed = (root / terminals_tts[0]["packed_file"]).read_bytes()
        require(sha(complete_packed) == terminals_tts[0]["packed_sha256"],
                "native TTS complete packed hash mismatch")
        complete = msgpack.unpackb(complete_packed, raw=False)
        wav_raw = complete["result"]["data"]
        require(sha(wav_raw) == terminals_tts[0]["wav_sha256"],
                "native TTS WAV hash mismatch")
        with wave.open(io.BytesIO(wav_raw), "rb") as wav:
            require(wav.getnchannels() == 1 and wav.getsampwidth() == 2
                    and wav.getframerate() == 24000, "invalid native TTS WAV")
            wav_pcm = wav.readframes(wav.getnframes())
        native_pcm = b"".join(native_pcm_parts)
        require(native_pcm == wav_pcm,
                f"public/native PCM concatenation differs from WAV: {child_id}")
        expected_join = report["tts_legs"][leg_index]
        require(expected_join["request_id"] == child_id
                and expected_join["chunk_count"] == len(native_chunks)
                and expected_join["pcm_bytes"] == len(native_pcm)
                and expected_join["pcm_sha256"] == sha(native_pcm)
                and expected_join["terminal_wav_sha256"] == sha(wav_raw)
                and expected_join["exact_native_public_wav_join"] is True,
                f"reported TTS join differs: {child_id}")
        verified_tts_legs.append({"request_id": child_id,
                                  "chunk_count": len(native_chunks),
                                  "pcm_sha256": sha(native_pcm),
                                  "terminal_wav_sha256": sha(wav_raw),
                                  "exact": True})
    require(public_cursor == len(by_media["audio"]),
            "extra public PCM chunks do not belong to any submitted leg")
    require(len(verified_tts_legs) == len(report["tts_legs"])
            == len(report["tts_submits"]) and bool(verified_tts_legs),
            "TTS leg join is incomplete")

    source_final_ns = report["source_final"]["observed_absolute_monotonic_ns"]
    require(video_submit["observed_monotonic_ns"]
            <= video_submit["forwarded_monotonic_ns"], "Wan tap ticks are nonmonotonic")
    processes = {item["name"]: item for item in report["stage_lifecycle"]["processes"]}
    require(processes["image"]["stop_monotonic_ns"] <= processes["tts"]["start_monotonic_ns"]
            <= processes["tts"]["stop_monotonic_ns"] <= processes["video"]["start_monotonic_ns"],
            "output stages overlapped")
    # The tap can queue the submit before the final source SSE reaches the
    # client. Native receipt/execution is instead bounded by the later Wan
    # process start, readiness marker, and native request log below.
    require(video_submit["forwarded_monotonic_ns"]
            <= processes["video"]["start_monotonic_ns"], "Wan tap forwarding was not queued")
    require(source_final_ns <= processes["video"]["start_monotonic_ns"]
            <= processes["video"]["ready_monotonic_ns"],
            "native Wan start/readiness predates public source completion")
    require(processes["vlm"]["start_monotonic_ns"] <= processes["image"]["start_monotonic_ns"]
            and processes["video"]["stop_monotonic_ns"] <= processes["vlm"]["stop_monotonic_ns"],
            "serial VLM was not resident across output sequencing")
    require(all(item["returncode"] == 0 for item in processes.values()),
            "a task-owned process did not stop cleanly")
    video_log = (root / report["native_video_log"]).read_text(encoding="utf-8", errors="replace")
    native_lines = [line for line in video_log.splitlines()
                    if "video request backend=native-cuda-wan" in line]
    require("native stage video ready kind=video" in video_log,
            "native Wan readiness marker is absent")
    require(len(native_lines) == 1 and " i2v=1 " in native_lines[0]
            and f"request_id={record['request_id']}:slot:1 " in native_lines[0]
            and " i2v=0 " not in video_log, "native Wan did not execute exactly one anchored I2V")
    tts_log = (root / report["native_tts_log"]).read_text(encoding="utf-8", errors="replace")
    require(tts_log.count("tts stream_check request_id=") == len(verified_tts_legs)
            and tts_log.count(" identical=1 ") == len(verified_tts_legs),
            "native streaming TTS byte-identity checks failed")
    require(report["health_final"].get("in_flight") == 0,
            "public gateway retained in-flight work")
    gpu = report.get("gpu", report.get("gpu1"))
    require(gpu["after_cleanup"]["memory_used_mib"] <= 16,
            f"GPU {physical_gpu} was not cleaned")
    monitor = json.loads((root / report["gpu_monitor"]).read_text(encoding="utf-8"))
    ticks = [sample["observed_monotonic_ns"] for sample in monitor]
    require(ticks == sorted(ticks)
            and all(sample["index"] == physical_gpu for sample in monitor),
            f"GPU {physical_gpu} monitor is nonmonotonic or observed another GPU")

    decoded = {}
    for label, path, selector, evaluation in (
            ("source_full", reconstructed / "source.png", None, False),
            ("source_eval", reconstructed / "source.png", None, True),
            ("first_full", reconstructed / "video.mp4", "select=eq(n\\,0)", False),
            ("last_full", reconstructed / "video.mp4", "select=eq(n\\,28)", False),
            ("first_eval", reconstructed / "video.mp4", "select=eq(n\\,0)", True),
            ("last_eval", reconstructed / "video.mp4", "select=eq(n\\,28)", True)):
        width, height, payload = rgb(args.ffmpeg, path, selector, evaluation)
        name = f"{label}.rgb{width}x{height}"
        (reconstructed / name).write_bytes(payload)
        identity = file_sha(reconstructed / name)
        expected = report["decoded_rgb"][label]
        require(identity["sha256"] == expected["sha256"]
                and identity["bytes"] == expected["bytes"]
                and (width, height) == (expected["width"], expected["height"]),
                f"decoded RGB identity mismatch: {label}")
        decoded[label] = {"file": name, **identity}
    first_mae = round(mae((reconstructed / decoded["source_eval"]["file"]).read_bytes(),
                          (reconstructed / decoded["first_eval"]["file"]).read_bytes()), 6)
    motion_mae = round(mae((reconstructed / decoded["first_eval"]["file"]).read_bytes(),
                           (reconstructed / decoded["last_eval"]["file"]).read_bytes()), 6)
    require(first_mae == report["first_frame_mae"] and motion_mae == report["motion_mae"],
            "independent RGB proxy values differ")
    require(first_mae <= report["first_frame_max_mae"]
            and motion_mae >= report["motion_min_mae"], "frozen RGB proxy gates failed")
    probe = subprocess.run([
        args.ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,nb_frames,r_frame_rate,pix_fmt", "-of", "json",
        str(reconstructed / "video.mp4")], check=True, capture_output=True, text=True, timeout=30)
    video_stream = json.loads(probe.stdout)["streams"][0]
    require(video_stream == report["video_stream"], "ffprobe result differs")

    provenance = {"skipped": args.skip_provenance_rehash}
    if not args.skip_provenance_rehash:
        binaries = {}
        for name in ("server_binary", "vlm_binary", "tts_binary", "wan_binary", "flux_stage",
                     "live_harness", "independent_verifier"):
            item = report["provenance"][name]
            actual = file_sha(item["path"])
            if name == "independent_verifier":
                # A corrected independent verifier may intentionally verify an
                # already sealed capture. Record both identities rather than
                # disguising the producer and verification code as one version.
                binaries[name] = {"producer": item, "verification": actual}
            else:
                require(actual["sha256"] == item["sha256"] and actual["bytes"] == item["bytes"],
                        f"external provenance changed for {name}")
                binaries[name] = actual
        models = {name: verify_model(identity) for name, identity in report["models"].items()}
        provenance = {"skipped": False, "binaries": binaries, "models": models}
    binding = {"source_sha256": sha(source_raw), "video_sha256": sha(video_raw),
               "anchor_sha256": inputs["anchor_sha256"], "native_binary_image_equal": True,
               "clean_tts_text_sha256": sha(tts_text.encode()), "audio": audio,
               "tts_legs": verified_tts_legs,
               "decoded_rgb": decoded, "first_frame_mae": first_mae, "motion_mae": motion_mae,
               "video_stream": video_stream, "provenance": provenance}
    (reconstructed / "binding.json").write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
    result = {"result": "PASS", "events": len(events), "artifacts": len(rebuilt),
              "artifact_chunks": chunk_total, "raw_sha256": sha(raw),
              "source_sha256": sha(source_raw), "video_sha256": sha(video_raw),
              "tts_submit_count": len(tts_text_parts), "pcm_count": len(audio),
              "tts_leg_count": len(verified_tts_legs),
              "exact_native_public_wav_join": True,
              "first_frame_mae": first_mae, "motion_mae": motion_mae,
              "fresh_reconstruction": str(reconstructed), "provenance": provenance,
              "perceptual_quality": "NOT_RUN", "generalization": "NOT_COVERED"}
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "provenance"},
                     sort_keys=True))
    print("INDEPENDENT_PUBLIC_NARRATIVE_RECONSTRUCTION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
