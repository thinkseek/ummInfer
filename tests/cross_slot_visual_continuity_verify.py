#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent fresh-process verifier for cross-slot visual continuity evidence.

Imports neither the live driver nor trajectory/replay producers. It parses raw
SSE, CAS, and MessagePack; rebuilds every public artifact; verifies the exact
image->image->I2V source chain, cancellation/wrong-reference controls, native
wrong-hash rejection/recovery, PCM/WAV/text joins, media geometry, process/GPU
lifecycle, and an optional self-excluded SHA256SUMS manifest.
"""
import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tempfile
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
        frame_raw = raw[cursor:end + 2]
        data = []
        for line in frame_raw[:-2].split(b"\n"):
            if line.startswith(b"data:"):
                value = line[5:]
                data.append(value[1:] if value.startswith(b" ") else value)
        require(data, "non-data frame is outside this capture contract")
        frames.append((cursor, frame_raw, b"\n".join(data)))
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
    return " ".join(text.split())


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


def verify_manifest(root):
    manifest = root / "SHA256SUMS"
    require(manifest.is_file(), "self-excluded manifest is absent")
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        require(len(digest) == 64 and name != "SHA256SUMS"
                and not name.startswith("run/") and name not in listed,
                f"invalid/duplicate/self-included manifest row: {name}")
        listed[name] = digest
    actual = sorted(str(path.relative_to(root)) for path in root.rglob("*")
                    if path.is_file() and path.name != "SHA256SUMS"
                    and "run" not in path.relative_to(root).parts)
    require(sorted(listed) == actual, "manifest does not cover the exact evidence file set")
    for name, digest in listed.items():
        require(file_sha(root / name)["sha256"] == digest,
                f"manifest digest mismatch: {name}")
    return {"files": len(listed), "manifest_sha256": file_sha(manifest)["sha256"]}


def read_capture(root, name, capture_desc):
    capture_root = root / name
    records = [json.loads(line) for line in
               (root / capture_desc["record"]).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    require(len(records) == 1, f"{name}: expected one trajectory")
    record = records[0]
    artifact_root = root / capture_desc["artifact_dir"]
    replay = record["sse_replay"]
    raw = (artifact_root / replay["raw_ref"]).read_bytes()
    require(sha(raw) == replay["raw_sha256"] and len(raw) == replay["raw_bytes"],
            f"{name}: raw SSE digest/bytes mismatch")
    frames = parse_frames(raw)
    require(len(frames) == len(replay["frames"]), f"{name}: frame count mismatch")
    events = []
    done = 0
    previous_tick = -1
    for index, (offset, frame_raw, payload) in enumerate(frames):
        desc = replay["frames"][index]
        require(desc["index"] == index and desc["offset"] == offset
                and desc["bytes"] == len(frame_raw) and desc["sha256"] == sha(frame_raw)
                and desc["tick"] >= previous_tick,
                f"{name}: raw frame binding/tick mismatch at {index}")
        previous_tick = desc["tick"]
        if payload == b"[DONE]":
            done += 1
            require(index == len(frames) - 1 and desc["kind"] == "done",
                    f"{name}: [DONE] missing or not final")
            continue
        require(desc["kind"] == "event" and desc["data_sha256"] == sha(payload),
                f"{name}: event descriptor mismatch at {index}")
        frame = json.loads(payload)
        if frame.get("unified") is None:
            require(frame.get("object") == "chat.completion.chunk"
                    and isinstance(frame.get("choices"), list),
                    f"{name}: invalid transport-only frame")
            continue
        events.append((index, frame["unified"]))
    require(done == 1 and len(events) == len(record["events"]),
            f"{name}: public event/[DONE] count mismatch")
    require([event.get("seq") for _, event in events] == list(range(len(events))),
            f"{name}: unified sequence has a gap")
    for event, (frame_index, raw_event) in zip(record["events"], events):
        require(event["source_event_index"] == frame_index
                and event["source_event_sha256"] == replay["frames"][frame_index]["sha256"]
                and event["source_event_tick"] == replay["frames"][frame_index]["tick"]
                and event["type"] == raw_event["kind"],
                f"{name}: normalized/raw event binding mismatch")
    terminals = [event for _, event in events
                 if event["kind"] in ("complete", "cancelled", "error")]
    require(len(terminals) == 1 and events[-1][1] is terminals[0],
            f"{name}: expected one final unified terminal")
    expected_kind = {"completed": "complete", "cancelled": "cancelled", "failed": "error"}[
        record["terminal_state"]]
    require(terminals[0]["kind"] == expected_kind
            and record["terminal_legs"] == terminals[0].get("legs", []),
            f"{name}: terminal state/legs mismatch")
    clean = "".join(event.get("text", "") for _, event in events
                    if event["kind"] == "vlm_delta")
    require(clean == record["clean_text"], f"{name}: clean text mismatch")
    require("[[" not in clean and "source=" not in clean,
            f"{name}: directive/source leaked into clean prose")
    slots = [event for _, event in events if event["kind"] == "visual_slot"]
    require(record["visual_slots"] == [
        {key: value for key, value in {
            "seq": event.get("seq"), "server_seq": event.get("seq"),
            "stage": event.get("stage"), "slot_id": event.get("slot_id"),
            "slot_index": event.get("slot_index"), "media_type": event.get("media_type"),
            "prompt": event.get("prompt"), "text_offset": event.get("text_offset"),
            "continuity_mode": event.get("continuity_mode"),
            "anchor_slot_index": event.get("anchor_slot_index"),
            "anchor_explicit": event.get("anchor_explicit"),
        }.items() if value is not None} for event in slots],
        f"{name}: visual_slot projection mismatch")
    require(record["session_id"] == record["request_params"].get("session_id"),
            f"{name}: public session identity mismatch")

    grouped = {}
    for frame_index, event in events:
        if event["kind"] != "artifact":
            continue
        desc = event["artifact"]
        grouped.setdefault(desc["artifact_id"], []).append((frame_index, event["seq"], desc))
    rebuilt = {}
    chunk_total = 0
    for artifact_id, chunks in grouped.items():
        chunk_total += len(chunks)
        chunks.sort(key=lambda item: item[2]["index"])
        total = chunks[0][2]["total"]
        require([item[2]["index"] for item in chunks] == list(range(total)),
                f"{name}: artifact chunk gap for {artifact_id}")
        payload = b"".join(base64.b64decode(item[2]["data"], validate=True)
                           for item in chunks)
        final = chunks[-1][2]
        require(final["bytes"] == len(payload) and final["sha256"] == sha(payload),
                f"{name}: artifact digest/bytes mismatch for {artifact_id}")
        rebuilt[artifact_id] = {"raw": payload, "final": final, "chunks": chunks}
    require(len(record["artifacts"]) == len(rebuilt),
            f"{name}: CAS artifact count mismatch")
    for item in record["artifacts"]:
        payload = (artifact_root / item["ref"]).read_bytes()
        require(sha(payload) == item["sha256"] and len(payload) == item["bytes"]
                and rebuilt[item["artifact_id"]]["raw"] == payload,
                f"{name}: CAS/raw payload mismatch for {item['artifact_id']}")
    return {"record": record, "events": events, "raw": raw, "rebuilt": rebuilt,
            "chunk_total": chunk_total, "artifact_root": artifact_root,
            "clean": clean, "slots": slots, "terminal": terminals[0]}


def packed_message(root, item):
    packed = (root / item["packed_file"]).read_bytes()
    require(sha(packed) == item["packed_sha256"], "packed MessagePack digest mismatch")
    import msgpack
    return msgpack.unpackb(packed, raw=False)


def self_check():
    raw = (b"data: {\"unified\":{\"seq\":0,\"kind\":\"complete\"}}\n\n"
           b"data: [DONE]\n\n")
    frames = parse_frames(raw)
    require(len(frames) == 2 and frames[1][2] == b"[DONE]",
            "deterministic raw SSE framing failed")
    with tempfile.TemporaryDirectory(prefix="umm-cross-slot-manifest-") as tmp:
        root = Path(tmp)
        (root / "payload.bin").write_bytes(b"payload")
        manifest = root / "SHA256SUMS"
        manifest.write_text(f"{sha(b'payload')}  payload.bin\n", encoding="utf-8")
        require(verify_manifest(root)["files"] == 1, "valid manifest rejected")
        (root / "payload.bin").write_bytes(b"tampered")
        try:
            verify_manifest(root)
        except AssertionError:
            pass
        else:
            raise AssertionError("manifest tampering was not rejected")
        (root / "payload.bin").write_bytes(b"payload")
        manifest.write_text(f"{sha(b'payload')}  payload.bin\n{sha(b'x')}  SHA256SUMS\n",
                            encoding="utf-8")
        try:
            verify_manifest(root)
        except AssertionError:
            pass
        else:
            raise AssertionError("self-included manifest was not rejected")
    print("OFFLINE_CROSS_SLOT_VERIFY: PASS (raw SSE; manifest tamper/self-exclusion)")
    print("REAL_GENERATION: NOT_RUN")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, nargs="?")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--skip-provenance-rehash", action="store_true")
    parser.add_argument("--check-manifest", action="store_true")
    parser.add_argument("--reconstruction-dir", type=Path)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    if args.evidence is None or args.out is None:
        parser.error("verification requires evidence and --out (or use --self-check)")
    root = args.evidence.resolve()
    manifest = verify_manifest(root) if args.check_manifest else {"checked": False}
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    require(report["format"] == "umminfer-cross-slot-visual-continuity-v1"
            and report["result"] == "CONTROLLED_LIVE_PASS"
            and report["physical_gpu"] == 7, "unknown report/wrong physical GPU")
    captures = {name: read_capture(root, name, desc)
                for name, desc in report["captures"].items()}
    require(set(captures) == {"wrong", "cancel", "main"}, "capture set mismatch")
    wrong, cancel, main = captures["wrong"], captures["cancel"], captures["main"]
    require(wrong["record"]["terminal_state"] == "completed"
            and len(wrong["slots"]) == 1
            and wrong["slots"][0].get("anchor_explicit") is True
            and not wrong["rebuilt"]
            and len(wrong["record"]["terminal_legs"]) == 1
            and wrong["record"]["terminal_legs"][0]["state"] == "failed"
            and report["wrong_reference"]["terminal_state"] == "completed"
            and report["wrong_reference"]["artifact_count"] == 0,
            "wrong/self reference generated media or did not fail its slot")
    cancel_legs = [leg for leg in cancel["record"]["terminal_legs"]
                   if leg.get("slot_id") is not None]
    require(cancel["record"]["terminal_state"] == "cancelled"
            and len(cancel["slots"]) == len(cancel_legs) == 3
            and all(leg["state"] == "aborted" for leg in cancel_legs)
            and not [item for item in cancel["rebuilt"].values()
                     if item["final"]["media_type"] in ("image", "video")]
            and report["cancellation"]["terminal_state"] == "cancelled"
            and report["cancellation"]["valid_retry_request_id"]
                == main["record"]["request_id"],
            "cancelled three-slot chain leaked media or retained non-aborted slots")

    record = main["record"]
    events = main["events"]
    require(record["terminal_state"] == "completed"
            and record["request_id"] == report["public_request_id"]
            and len(events) == report["public_event_count"]
            and main["chunk_total"] == report["public_artifact_chunk_count"]
            and main["clean"] == report["clean_text"]
            and sha(main["clean"].encode()) == report["clean_text_sha256"],
            "main public report/raw capture mismatch")
    require(sha(main["raw"]) == report["raw_sse"]["raw_sha256"]
            and len(main["raw"]) == report["raw_sse"]["raw_bytes"],
            "main raw SSE summary mismatch")
    slots = main["slots"]
    require(len(slots) == 3
            and [slot["stage"] for slot in slots] == ["image", "image", "video"]
            and [slot["slot_index"] for slot in slots] == [0, 1, 2]
            and slots[1].get("anchor_slot_index") == 0
            and slots[2].get("anchor_slot_index") == 1
            and slots[1].get("anchor_explicit") is True
            and slots[2].get("anchor_explicit") is True
            and all("anchor_sha256" not in slot for slot in slots)
            and all("source=" not in slot["prompt"] for slot in slots),
            "main three-slot linkage metadata mismatch")
    by_slot = {}
    for item in main["rebuilt"].values():
        slot_id = item["final"].get("slot_id")
        if slot_id is not None:
            require(slot_id not in by_slot, f"duplicate visual artifact for {slot_id}")
            by_slot[slot_id] = item
    require(set(by_slot) == {"slot-0", "slot-1", "slot-2"}, "visual artifact set mismatch")
    source0 = by_slot["slot-0"]["raw"]
    source1 = by_slot["slot-1"]["raw"]
    video = by_slot["slot-2"]["raw"]
    require(by_slot["slot-0"]["final"]["media_type"] == "image"
            and by_slot["slot-1"]["final"]["media_type"] == "image"
            and by_slot["slot-2"]["final"]["media_type"] == "video",
            "visual artifact modalities mismatch")
    require(by_slot["slot-1"]["final"].get("anchor_slot_index") == 0
            and by_slot["slot-1"]["final"].get("anchor_explicit") is True
            and by_slot["slot-1"]["final"].get("anchor_sha256") == sha(source0)
            and by_slot["slot-2"]["final"].get("anchor_slot_index") == 1
            and by_slot["slot-2"]["final"].get("anchor_explicit") is True
            and by_slot["slot-2"]["final"].get("anchor_sha256") == sha(source1),
            "public artifact content-addressed chain mismatch")
    require(by_slot["slot-1"]["chunks"][0][1] > by_slot["slot-0"]["chunks"][-1][1]
            and by_slot["slot-2"]["chunks"][0][1] > by_slot["slot-1"]["chunks"][-1][1],
            "dependent public artifact predates its selected source")
    legs = {leg.get("slot_id"): leg for leg in record["terminal_legs"]
            if leg.get("slot_id") is not None}
    require(set(legs) == {"slot-0", "slot-1", "slot-2"}
            and all(leg["state"] == "completed" for leg in legs.values())
            and legs["slot-1"].get("anchor_sha256") == sha(source0)
            and legs["slot-2"].get("anchor_sha256") == sha(source1)
            and any(leg.get("stage") == "tts" and leg.get("state") == "completed"
                    for leg in record["terminal_legs"]),
            "main terminal visual/TTS lifecycle mismatch")
    expected_chain = [
        {"dependent_slot": 1, "source_slot": 0, "source_sha256": sha(source0)},
        {"dependent_slot": 2, "source_slot": 1, "source_sha256": sha(source1)},
    ]
    require(report["reference_chain"] == expected_chain, "reported reference chain differs")

    import msgpack
    image_submits = []
    for path in sorted(root.glob("image-*.msgpack")):
        if path.name.startswith("image-output-"):
            continue
        message = msgpack.unpackb(path.read_bytes(), raw=False)
        if message.get("type") == "submit":
            image_submits.append((path, message))
    video_submits = []
    for path in sorted(root.glob("video-*.msgpack")):
        message = msgpack.unpackb(path.read_bytes(), raw=False)
        if message.get("type") == "submit":
            video_submits.append((path, message))
    cancel_prefix = cancel["record"]["request_id"] + ":slot:"
    main_prefix = record["request_id"] + ":slot:"
    cancel_images = [item for item in image_submits
                     if item[1].get("request_id", "").startswith(cancel_prefix)]
    main_images = [item for item in image_submits
                   if item[1].get("request_id", "").startswith(main_prefix)]
    require(len(cancel_images) == 1
            and cancel_images[0][1]["request_id"].endswith(":slot:0")
            and not [item for item in video_submits
                     if item[1].get("request_id", "").startswith(
                         cancel["record"]["request_id"] + ":")],
            "cancelled request dispatched a transitive visual dependent")
    require(len(main_images) == 2 and len(video_submits) == 1,
            "main native visual submit count mismatch")
    main_images.sort(key=lambda item: item[1]["request_id"])
    root_inputs = main_images[0][1]["data"]["request"]["inputs"]
    edit_inputs = main_images[1][1]["data"]["request"]["inputs"]
    require(main_images[0][1]["request_id"].endswith(":slot:0")
            and "image" not in root_inputs
            and main_images[1][1]["request_id"].endswith(":slot:1")
            and edit_inputs.get("image") == source0
            and edit_inputs.get("anchor_sha256") == sha(source0)
            and edit_inputs.get("anchor_slot_index") == 0
            and edit_inputs.get("anchor_explicit") is True
            and "anchor_crop" not in edit_inputs
            and "source=" not in edit_inputs["prompt"],
            "FLUX edit did not consume exact slot-0 bytes/content address")
    video_path, video_message = video_submits[0]
    require(video_message["request_id"] == record["request_id"] + ":slot:2",
            "Wan request is not main slot-2")
    video_inputs = video_message["data"]["request"]["inputs"]
    require(video_inputs.get("image") == source1
            and video_inputs.get("anchor_sha256") == sha(source1)
            and video_inputs.get("anchor_slot_index") == 1
            and video_inputs.get("anchor_explicit") is True
            and video_inputs.get("anchor_crop") == "center_cover"
            and "source=" not in video_inputs["prompt"],
            "Wan did not consume exact slot-1 bytes/content address")

    native_results = []
    for control in report["native_controls"]:
        submit_raw = (root / control["submit_file"]).read_bytes()
        require(sha(submit_raw) == control["submit_sha256"], "native control submit hash")
        submit = msgpack.unpackb(submit_raw, raw=False)
        inputs = submit["data"]["request"]["inputs"]
        require(submit["request_id"] == control["request_id"]
                and inputs["image"] == source0
                and sha(inputs["image"]) == control["source_sha256"] == sha(source0)
                and inputs["anchor_explicit"] is True
                and inputs["anchor_slot_index"] == 0
                and inputs["anchor_sha256"] == control["declared_anchor_sha256"],
                "native control source/declared hash mismatch")
        output_raw = (root / control["output_file"]).read_bytes()
        require(sha(output_raw) == control["output_packed_sha256"],
                "native control output MessagePack hash mismatch")
        complete = msgpack.unpackb(output_raw, raw=False)
        require(complete["type"] == "complete"
                and complete["request_id"] == control["request_id"]
                and complete["success"] is control["success"],
                "native control terminal mismatch")
        if control["request_id"] == "native-76-wrong-hash":
            require(control["success"] is False
                    and control["declared_anchor_sha256"] == "0" * 64
                    and "does not match anchor_sha256" in complete["error"],
                    "wrong native source hash was not rejected")
        else:
            require(control["request_id"] == "native-76-valid-retry"
                    and control["success"] is True
                    and control["declared_anchor_sha256"] == sha(source0),
                    "valid native recovery did not complete")
            png = complete["result"]["data"]
            require(sha(png) == control["output_sha256"]
                    and len(png) == control["output_bytes"]
                    and png.startswith(b"\x89PNG\r\n\x1a\n")
                    and png == (root / "native-control-recovery.png").read_bytes(),
                    "native valid recovery PNG mismatch")
            native_results.append(png)
    require(len(report["native_controls"]) == 2 and len(native_results) == 1,
            "native wrong-hash/valid-recovery pair is incomplete")

    reconstructed = args.reconstruction_dir or root / "reconstructed"
    require(not reconstructed.exists(), "reconstruction directory already exists")
    reconstructed.mkdir(parents=True)
    for name, payload in (("slot-0.png", source0), ("slot-1.png", source1),
                          ("slot-2.mp4", video),
                          ("native-control-recovery.png", native_results[0]),
                          ("raw.sse", main["raw"])):
        (reconstructed / name).write_bytes(payload)
    (reconstructed / "events.json").write_text(json.dumps(
        [event for _, event in events], indent=2, sort_keys=True) + "\n")

    by_media = {}
    for item in main["rebuilt"].values():
        by_media.setdefault(item["final"]["media_type"], []).append(item)
    audio = []
    for index, item in enumerate(by_media.get("audio", [])):
        pcm = item["raw"]
        desc = item["final"]
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
    require(audio and len(audio) == len(report["audio"]), "speech artifact count mismatch")

    tts_text_parts = []
    for submit in report["tts_submits"]:
        message = packed_message(root, submit)
        text = message["data"]["request"]["inputs"]["text"]
        require(text == submit["text"] and sha(text.encode()) == submit["text_sha256"],
                "TTS text submit identity mismatch")
        tts_text_parts.append(text)
    tts_text = " ".join(tts_text_parts)
    require(spoken_projection(tts_text) == spoken_projection(main["clean"])
            and "[[" not in tts_text and "source=" not in tts_text,
            "TTS text differs from clean prose or leaks a directive")
    public_audio = by_media.get("audio", [])
    public_cursor = 0
    verified_tts = []
    for leg_index, submit in enumerate(report["tts_submits"]):
        child_id = submit["request_id"]
        chunks = [item for item in report["tts_output"]
                  if item.get("message_type") == "stream"
                  and item.get("request_id") == child_id]
        terminals = [item for item in report["tts_output"]
                     if item.get("message_type") == "complete"
                     and item.get("request_id") == child_id]
        require(chunks and len(terminals) == 1 and terminals[0]["success"] is True
                and terminals[0]["streamed"] is True, "incomplete native TTS leg")
        native_parts = []
        for expected_index, native in enumerate(chunks):
            message = packed_message(root, native)
            native_pcm = message["chunk"]["audio"]
            require(message["request_id"] == child_id
                    and message["chunk"]["index"] == expected_index
                    and sha(native_pcm) == native["audio_sha256"]
                    and public_cursor < len(public_audio)
                    and public_audio[public_cursor]["raw"] == native_pcm
                    and public_audio[public_cursor]["final"].get("stream_index")
                        == expected_index,
                    "public/native PCM chunk join mismatch")
            native_parts.append(native_pcm)
            public_cursor += 1
        complete = packed_message(root, terminals[0])
        wav_raw = complete["result"]["data"]
        with wave.open(io.BytesIO(wav_raw), "rb") as wav:
            require(wav.getnchannels() == 1 and wav.getsampwidth() == 2
                    and wav.getframerate() == 24000, "invalid native TTS WAV")
            wav_pcm = wav.readframes(wav.getnframes())
        joined = b"".join(native_parts)
        require(joined == wav_pcm and sha(wav_raw) == terminals[0]["wav_sha256"],
                "native/public PCM concatenation differs from terminal WAV")
        expected = report["tts_legs"][leg_index]
        require(expected["request_id"] == child_id
                and expected["chunk_count"] == len(chunks)
                and expected["pcm_sha256"] == sha(joined)
                and expected["terminal_wav_sha256"] == sha(wav_raw)
                and expected["exact_native_public_wav_join"] is True,
                "reported TTS join differs")
        verified_tts.append({"request_id": child_id, "chunks": len(chunks),
                             "pcm_sha256": sha(joined), "exact": True})
    require(public_cursor == len(public_audio) and verified_tts,
            "unowned public PCM remains or no TTS leg was verified")

    processes = {item["name"]: item for item in report["stage_lifecycle"]["processes"]}
    require(set(processes) == {"server", "entry", "vlm", "image", "tts", "video"}
            and all(item["returncode"] == 0 for item in processes.values()),
            "a task-owned process is missing or did not stop cleanly")
    edited_ns = report["source_observations"]["slot-1"]["observed_absolute_monotonic_ns"]
    require(edited_ns <= report["stage_lifecycle"]["edited_public_complete_ns"]
            <= processes["image"]["stop_monotonic_ns"]
            <= processes["video"]["start_monotonic_ns"]
            <= processes["video"]["ready_monotonic_ns"],
            "FLUX stop/Wan start does not follow selected edited-source completion")
    require(processes["vlm"]["start_monotonic_ns"] <= processes["image"]["start_monotonic_ns"]
            and processes["video"]["stop_monotonic_ns"] <= processes["vlm"]["stop_monotonic_ns"],
            "VLM was not resident across the complete public gate")
    video_log = (root / report["native_video_log"]).read_text(encoding="utf-8", errors="replace")
    native_lines = [line for line in video_log.splitlines()
                    if "video request backend=native-cuda-wan" in line]
    require(len(native_lines) == 1 and " i2v=1 " in native_lines[0]
            and f"request_id={record['request_id']}:slot:2 " in native_lines[0]
            and " i2v=0 " not in video_log,
            "native Wan did not execute exactly one slot-1-anchored I2V")
    image_log = (root / report["native_image_log"]).read_text(encoding="utf-8", errors="replace")
    require("request_id=native-76-wrong-hash status=error" in image_log
            and "does not match anchor_sha256" in image_log
            and "request_id=native-76-wrong-hash status=ok" not in image_log
            and "request_id=native-76-valid-retry status=ok" in image_log,
            "native wrong-hash rejection/valid recovery logs are incomplete")
    tts_log = (root / report["native_tts_log"]).read_text(encoding="utf-8", errors="replace")
    require(all(f"tts stream_check request_id={item['request_id']}" in tts_log
                for item in verified_tts), "native TTS stream checks are incomplete")
    require(report["health_final"].get("in_flight") == 0
            and report["gpu7"]["after_cleanup"]["memory_used_mib"] <= 16
            and report["gpu7"]["after_cleanup"]["utilization_percent"] == 0
            and report["gpu7"]["after_cleanup"].get("compute_processes") == 0,
            "public in-flight work or GPU 7 memory/utilization remained")
    monitor = json.loads((root / report["gpu_monitor"]).read_text(encoding="utf-8"))
    ticks = [sample["observed_monotonic_ns"] for sample in monitor]
    require(ticks == sorted(ticks) and bool(monitor)
            and all(sample["index"] == 7 for sample in monitor),
            "GPU 7 monitor is empty, nonmonotonic, or observed another GPU")

    decoded = {}
    for label, path, selector, evaluation in (
            ("source0_full", reconstructed / "slot-0.png", None, False),
            ("source1_full", reconstructed / "slot-1.png", None, False),
            ("source1_eval", reconstructed / "slot-1.png", None, True),
            ("first_full", reconstructed / "slot-2.mp4", "select=eq(n\\,0)", False),
            ("last_full", reconstructed / "slot-2.mp4", "select=eq(n\\,28)", False),
            ("first_eval", reconstructed / "slot-2.mp4", "select=eq(n\\,0)", True),
            ("last_eval", reconstructed / "slot-2.mp4", "select=eq(n\\,28)", True),
            ("native_recovery_full", reconstructed / "native-control-recovery.png",
             None, False)):
        width, height, payload = rgb(args.ffmpeg, path, selector, evaluation)
        name = f"{label}.rgb{width}x{height}"
        (reconstructed / name).write_bytes(payload)
        identity = file_sha(reconstructed / name)
        expected = report["decoded_rgb"][label]
        require(identity["sha256"] == expected["sha256"]
                and identity["bytes"] == expected["bytes"]
                and (width, height) == (expected["width"], expected["height"]),
                f"decoded RGB identity mismatch: {label}")
        decoded[label] = {"file": name, "width": width, "height": height, **identity}
    for label in ("source0_full", "source1_full", "native_recovery_full"):
        require((decoded[label]["width"], decoded[label]["height"]) == (512, 512),
                f"unexpected decoded source dimensions: {label}")
    first_mae = round(mae((reconstructed / decoded["source1_eval"]["file"]).read_bytes(),
                          (reconstructed / decoded["first_eval"]["file"]).read_bytes()), 6)
    motion_mae = round(mae((reconstructed / decoded["first_eval"]["file"]).read_bytes(),
                           (reconstructed / decoded["last_eval"]["file"]).read_bytes()), 6)
    require(first_mae == report["first_frame_mae"]
            and motion_mae == report["motion_mae"]
            and first_mae <= report["first_frame_max_mae"]
            and motion_mae >= report["motion_min_mae"],
            "independent continuity/motion proxy mismatch or gate failure")
    probe = subprocess.run([
        args.ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,nb_frames,r_frame_rate,pix_fmt", "-of", "json",
        str(reconstructed / "slot-2.mp4")], check=True, capture_output=True,
        text=True, timeout=30)
    video_stream = json.loads(probe.stdout)["streams"][0]
    require(video_stream == report["video_stream"]
            and video_stream["codec_name"] == "h264"
            and video_stream["pix_fmt"] == "yuv420p"
            and (video_stream["width"], video_stream["height"]) == (704, 480)
            and int(video_stream["nb_frames"]) == 29
            and video_stream["r_frame_rate"] == "16/1",
            "decoded video geometry/codec/frame contract differs")

    provenance = {"skipped": args.skip_provenance_rehash}
    if not args.skip_provenance_rehash:
        binaries = {}
        for name in ("server_binary", "vlm_binary", "tts_binary", "wan_binary",
                     "flux_stage", "live_harness", "shared_live_helpers",
                     "independent_verifier"):
            item = report["provenance"][name]
            actual = file_sha(item["path"])
            if name == "independent_verifier":
                # A verifier-only correction may inspect an already sealed raw
                # capture. Preserve producer and verification source identities
                # separately rather than aliasing them or changing the media.
                binaries[name] = {"producer": item, "verification": actual}
            else:
                require(actual["sha256"] == item["sha256"]
                        and actual["bytes"] == item["bytes"],
                        f"external provenance changed for {name}")
                binaries[name] = actual
        models = {name: verify_model(identity)
                  for name, identity in report["models"].items()}
        provenance = {"skipped": False, "binaries": binaries, "models": models}
    result = {
        "result": "PASS", "events": len(events), "artifacts": len(main["rebuilt"]),
        "artifact_chunks": main["chunk_total"], "raw_sha256": sha(main["raw"]),
        "reference_chain": expected_chain,
        "source0_sha256": sha(source0), "source1_sha256": sha(source1),
        "video_sha256": sha(video), "native_binary_source_equal": True,
        "wrong_reference_rejected": True, "cancelled_dependents_aborted": 3,
        "native_wrong_hash_rejected": True, "native_valid_recovery": True,
        "tts_leg_count": len(verified_tts), "pcm_count": len(audio),
        "exact_native_public_wav_join": True, "clean_text_join": True,
        "first_frame_mae": first_mae, "motion_mae": motion_mae,
        "source_dimensions": "512x512,512x512", "video_stream": video_stream,
        "decoded_rgb": decoded, "manifest": manifest, "provenance": provenance,
        "perceptual_quality": "NOT_RUN", "human_review": "NOT_RUN",
        "generalization": "NOT_COVERED", "resident_service": "NOT_CONTACTED",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("provenance", "decoded_rgb")}, sort_keys=True))
    print("INDEPENDENT_CROSS_SLOT_RECONSTRUCTION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
