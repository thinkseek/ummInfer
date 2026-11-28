#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent fresh-process verifier for the NS-01 streaming-opening bundle.

No producer module is imported. Raw SSE, CAS, MessagePack, WAV, logs, binary/model
identities, pair aggregates, decisions, and the complete manifest are recomputed.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import wave
import msgpack


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def ms(value):
    return None if value is None else round(value, 3)


def median(values):
    values = [value for value in values if value is not None]
    return None if not values else round(statistics.median(values), 3)


def request_body(prereg, case_name):
    case = prereg["golden_cases"][case_name]
    request = prereg["request"]
    return {
        "model": "qwen3-native", "stream": request["stream"],
        "messages": [{"role": "user", "content": case["user_text"]}],
        "modalities": request["modalities"], "image": request["image"],
        "audio": {key: value for key, value in request["audio"].items()
                  if key in ("voice", "format", "stream")},
        "video": request["video"], "narrative": request["narrative"],
        "max_tokens": request["max_tokens"], "temperature": request["temperature"],
        "seed": request["seed"], "session_id": case["scenario"],
    }


def parse_sse(raw):
    frames = []
    offset = 0
    for block in raw.split(b"\n\n"):
        if not block:
            continue
        lines = block.splitlines()
        data = b"\n".join(line[5:].lstrip() for line in lines if line.startswith(b"data:"))
        require(data, "SSE frame lacks data")
        raw_frame = block + b"\n\n"
        frames.append({"offset": offset, "raw": raw_frame, "data": data,
                       "sha256": sha(raw_frame), "data_sha256": sha(data)})
        offset += len(raw_frame)
    require(offset == len(raw), "SSE framing did not reconstruct exact raw bytes")
    return frames


def native_tts_logs(path):
    result = {}
    pattern = re.compile(
        r"tts request request_id=(\S+).*?codec_frames=(\d+) natural_eos=(\d+)"
        r" samples=(\d+).*?elapsed_ms=(\d+)")
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            match = pattern.search(line)
            if match:
                result[match.group(1)] = {
                    "codec_frames": int(match.group(2)), "natural_eos": match.group(3) == "1",
                    "samples": int(match.group(4)), "native_elapsed_ms": int(match.group(5))}
    return result


def vlm_finish(path):
    matches = list(re.finditer(
        r"qwen38 request request_id=\S+.*?generated_tokens=(\d+) finish_reason=(\S+)",
        path.read_text(errors="replace") if path.exists() else ""))
    if not matches:
        return {"finish_reason": "MISSING", "generated_tokens": None}
    match = matches[-1]
    return {"generated_tokens": int(match.group(1)), "finish_reason": match.group(2)}


def content_projection(text, slots, case_name, prereg):
    case = prereg["golden_cases"][case_name]
    headings = list(re.finditer(r"(?m)^ {0,3}#{1,6}[ \t]+", text))
    first_heading = headings[0].start() if headings else len(text)
    before = text[:first_heading]
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", before) if part.strip()]
    image = next((slot for slot in slots if slot.get("slot_id") == "slot-0"), None)
    video = next((slot for slot in slots if slot.get("slot_id") == "slot-1"), None)
    opening_bytes = int(image.get("text_offset", 0)) if image else 0
    opening = text.encode()[:opening_bytes].decode(errors="ignore").strip()
    chinese = len(re.findall(r"[㐀-鿿]", before))
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", before)
    opening_words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", opening)
    opening_chinese = len(re.findall(r"[㐀-鿿]", opening))
    units = opening_chinese if case_name == "chinese" else len(opening_words)
    terms = [term for term in case["required_terms"] if term.lower() in text.lower()]
    return {
        "clean_text_bytes": len(text.encode()), "clean_text_sha256": sha(text.encode()),
        "heading_count": len(headings), "opening_paragraphs_before_heading": len(paragraphs),
        "opening_chinese_codepoints_before_heading": chinese,
        "opening_english_words_before_heading": len(words), "opening": opening,
        "opening_units": units, "opening_before_image": bool(opening) and opening_bytes > 0,
        "opening_is_concise": 0 < units <= prereg["content_and_lifecycle_gates"]
                                               ["opening_max_words_or_chinese_codepoints"],
        "opening_has_no_heading": not re.search(r"(?m)^\s*#", opening), "term_hits": terms,
        "tag_leaks": len(re.findall(r"\[\[\s*(?:image|video|visual)\s*:|source=slot-\d+",
                                    text, flags=re.I)),
        "image_slot_present": image is not None, "video_slot_present": video is not None,
        "video_explicit_anchor": bool(video and video.get("anchor_explicit") is True
                                      and video.get("anchor_slot_index") == 0),
        "minimum_clean_text_bytes_met": len(text.encode()) >= case["minimum_clean_text_bytes"],
        "minimum_opening_paragraphs_met": len(paragraphs) >=
                                          case["minimum_opening_paragraphs_before_first_heading"],
        "minimum_background_met": chinese >= case.get(
            "minimum_opening_chinese_codepoints_before_first_heading", 0) and len(words) >=
            case.get("minimum_opening_english_words_before_first_heading", 0),
        "minimum_headings_met": len(headings) >= case["minimum_markdown_heading_lines"],
        "minimum_terms_met": len(terms) >= case["minimum_required_terms"],
    }


def spoken_text_projection(text):
    text = re.sub(r"(?m)^ {0,3}(?:#+[ \t]+|>[ \t]?|[*+-][ \t]+)", "", text)
    text = re.sub(r"(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", "", text)
    out = []
    states = {"*": False, "_": False}
    ascii_space = lambda c: c in " \t\r\n"
    ascii_alnum = lambda c: c.isascii() and c.isalnum()
    i = 0
    while i < len(text):
        marker = text[i]
        if marker not in "*_~`":
            out.append(marker)
            i += 1
            continue
        end = i + 1
        while end < len(text) and text[end] == marker:
            end += 1
        count = end - i
        remove = marker == "`" or (marker == "~" and count >= 2) or \
            (marker in "*_" and count >= 2)
        left_alnum = i > 0 and ascii_alnum(text[i - 1])
        right_alnum = end < len(text) and ascii_alnum(text[end])
        if marker == "_" and count >= 2:
            remove = not (left_alnum and right_alnum)
        if marker in "*_" and count == 1:
            left_space = i == 0 or ascii_space(text[i - 1])
            right_space = end == len(text) or ascii_space(text[end])
            digit_arithmetic = i > 0 and end < len(text) and text[i - 1].isdigit() \
                and text[end].isdigit()
            intraword_underscore = marker == "_" and left_alnum and right_alnum
            can_open = not right_space and not digit_arithmetic and not intraword_underscore
            can_close = not left_space or (i == 0 and states[marker])
            if not digit_arithmetic and not intraword_underscore and \
                    ((states[marker] and can_close) or (not states[marker] and can_open)):
                states[marker] = not states[marker]
                remove = True
            else:
                remove = False
        if not remove:
            out.append(text[i:end])
        i = end
    return "".join("".join(out).split())


def native_speech_has_presentation_tokens(text):
    return bool(re.search(r"(?:\*\*|__|~~|`)", text) or
                re.search(r"(?m)^ {0,3}#+[ \t]+", text) or
                re.search(r"(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", text))


def verify_arm(bundle, name, expected, prereg, candidate, identity_cache):
    arm_dir = bundle / name
    arm = load(arm_dir / "arm.json")
    record = load(arm_dir / "public-record.json")
    provenance = load(arm_dir / "provenance.json")
    case_name, repetition_text, arm_name = name.rsplit("-", 2)[0], name.rsplit("-", 2)[1], name.rsplit("-", 2)[2]
    repetition = int(repetition_text[1:])
    require((arm["case"], arm["repetition"], arm["arm"]) == (case_name, repetition, arm_name),
            f"arm identity mismatch: {name}")
    require(expected["case"] == case_name and expected["repetition"] == repetition
            and expected["arm"] == arm_name, f"scorecard identity mismatch: {name}")
    body = request_body(prereg, case_name)
    require(load(arm_dir / "request.json") == body, f"frozen request reconstruction failed: {name}")
    require(sha(canonical(body)) == arm["request_sha256"] == expected["request_sha256"],
            f"request hash mismatch: {name}")
    require(arm["execution_commit"] == expected["execution_commit"], f"commit mismatch: {name}")
    if arm_name == "after":
        require(arm["execution_commit"] == candidate["execution_commit"],
                f"after arm does not execute frozen candidate: {name}")
    for binary_name in ("server_binary", "vlm_binary", "tts_binary", "wan_binary"):
        item = provenance[binary_name]
        digest = identity_cache["binaries"].get(item["path"])
        if digest is None:
            digest = file_sha(item["path"])
            identity_cache["binaries"][item["path"]] = digest
        require(digest == item["sha256"] and Path(item["path"]).stat().st_size == item["bytes"],
                f"binary identity mismatch: {name}:{binary_name}")
        require(expected["binary_identity"][binary_name] == {
            "sha256": item["sha256"], "bytes": item["bytes"]}, f"binary projection mismatch: {name}")
    models = load(arm_dir / "model-identity.json")
    if not identity_cache["models"]:
        identity_cache["models"] = models
        for model_name, model in models.items():
            root = Path(model["root"])
            files = {}
            for path in sorted(p for p in root.rglob("*") if p.is_file()):
                relative = path.relative_to(root).as_posix()
                expected_file = model["files"][relative]
                actual = {"sha256": file_sha(path), "bytes": path.stat().st_size}
                require(actual == expected_file, f"model file mismatch: {model_name}:{relative}")
                files[relative] = actual
            require(set(files) == set(model["files"]), f"model file set mismatch: {model_name}")
            require(sha(canonical(files)) == model["tree_sha256"], f"model tree mismatch: {model_name}")
    require(models == identity_cache["models"], f"model identity differs across arms: {name}")
    model_projection = {}
    for model_name, model in models.items():
        config_name = "config.json" if "config.json" in model["files"] else "model_index.json"
        model_projection[model_name] = {
            "model_id": model["model_id"], "revision": "UNKNOWN",
            "tree_sha256": model["tree_sha256"], "file_count": len(model["files"]),
            "config_sha256": model["files"][config_name]["sha256"]}
    require(model_projection == expected["model_identity"], f"model identity projection mismatch: {name}")
    manifest = {}
    for line in (arm_dir / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        manifest[relative] = digest
    actual_files = {p.relative_to(arm_dir).as_posix(): p for p in arm_dir.rglob("*")
                    if p.is_file() and p.name != "SHA256SUMS" and "run" not in p.relative_to(arm_dir).parts}
    require(set(manifest) == set(actual_files), f"nested manifest file set mismatch: {name}")
    for relative, path in actual_files.items():
        require(file_sha(path) == manifest[relative], f"nested manifest mismatch: {name}:{relative}")
    require(file_sha(arm_dir / "SHA256SUMS") == expected["raw"]["arm_sha256sums_sha256"],
            f"nested manifest digest mismatch: {name}")

    replay = record["sse_replay"]
    raw = (arm_dir / "capture" / "artifacts" / replay["raw_ref"]).read_bytes()
    require(sha(raw) == replay["raw_sha256"] == expected["raw"]["raw_sse_sha256"],
            f"raw SSE hash mismatch: {name}")
    require(len(raw) == replay["raw_bytes"] == expected["raw"]["raw_sse_bytes"],
            f"raw SSE byte count mismatch: {name}")
    frames = parse_sse(raw)
    require(len(frames) == len(replay["frames"]), f"raw frame count mismatch: {name}")
    for index, (actual, bound) in enumerate(zip(frames, replay["frames"])):
        require(index == bound["index"] and actual["offset"] == bound["offset"]
                and len(actual["raw"]) == bound["bytes"] and actual["sha256"] == bound["sha256"]
                and actual["data_sha256"] == bound["data_sha256"], f"raw frame binding mismatch: {name}:{index}")
    public_events = []
    done = 0
    for index, frame in enumerate(frames):
        if frame["data"] == b"[DONE]":
            done += 1
            continue
        payload = json.loads(frame["data"])
        unified = payload.get("unified")
        if unified is None:
            continue
        unified = dict(unified)
        unified["_frame_index"] = index
        unified["_tick"] = replay["frames"][index]["tick"]
        public_events.append(unified)
    require(done == 1, f"DONE count is not one: {name}")
    require([event["seq"] for event in public_events] == list(range(len(public_events))),
            f"unified sequence gap: {name}")
    terminals = [event for event in public_events
                 if event.get("kind") in ("complete", "error", "cancelled")]
    require(len(terminals) == 1 and public_events[-1] is terminals[0],
            f"public terminal count/order mismatch: {name}")
    terminal_state = {"complete": "completed", "error": "error", "cancelled": "cancelled"}[
        terminals[0]["kind"]]
    require(terminal_state == record["terminal_state"] == arm["terminal_state"],
            f"public terminal state mismatch: {name}")
    clean = "".join(event.get("text", "") for event in public_events
                    if event.get("kind") == "vlm_delta")
    require(clean == record["clean_text"] == arm["clean_text"], f"clean text mismatch: {name}")
    require(sha(clean.encode()) == record["clean_text_sha256"] == expected["content"]["clean_text_sha256"],
            f"clean text digest mismatch: {name}")
    slots = [event for event in public_events if event.get("kind") == "visual_slot"]
    bound_slots = record["visual_slots"]
    require(len(slots) == len(bound_slots), f"slot count mismatch: {name}")
    for actual, bound in zip(slots, bound_slots):
        for key, value in bound.items():
            if key == "type":
                require(value == "visual_slot", f"slot type mismatch: {name}")
            elif key == "server_seq":
                require(actual["seq"] == value, f"slot server sequence mismatch: {name}")
            elif key == "source_event_index":
                require(actual["_frame_index"] == value, f"slot source frame mismatch: {name}")
            elif key == "source_event_sha256":
                require(frames[actual["_frame_index"]]["sha256"] == value,
                        f"slot source hash mismatch: {name}")
            elif key == "source_event_tick":
                require(actual["_tick"] == value, f"slot source tick mismatch: {name}")
            else:
                require(actual.get(key) == value, f"slot field mismatch: {name}:{key}")
    content = content_projection(clean, bound_slots, case_name, prereg)
    require(content == expected["content"], f"content score mismatch: {name}")

    first = {key: None for key in (
        "ttft_ms", "last_vlm_text_ms", "first_tts_submit_ms", "ttfa_ms", "image_slot_ms",
        "first_image_ms", "video_slot_ms", "i2v_native_submit_ms",
        "i2v_first_public_chunk_ms", "i2v_playable_ms", "tts_ready_ms", "wan_ready_ms")}
    for event in public_events:
        kind = event["kind"]
        tick_ms = event["_tick"] / 1e6
        if kind == "vlm_delta" and event.get("text"):
            if first["ttft_ms"] is None:
                first["ttft_ms"] = tick_ms
            first["last_vlm_text_ms"] = tick_ms
        elif kind == "visual_slot":
            if event.get("slot_id") == "slot-0" and first["image_slot_ms"] is None:
                first["image_slot_ms"] = tick_ms
            if event.get("slot_id") == "slot-1" and first["video_slot_ms"] is None:
                first["video_slot_ms"] = tick_ms
        elif kind == "artifact":
            desc = event["artifact"]
            raw_chunk = base64.b64decode(desc["data"], validate=True)
            require(sha(raw_chunk) == desc.get("chunk_sha256", sha(raw_chunk)),
                    f"raw artifact chunk hash mismatch: {name}")
            if desc["media_type"] == "audio":
                if first["ttfa_ms"] is None:
                    first["ttfa_ms"] = tick_ms
            elif desc["media_type"] == "image" and desc.get("slot_id") == "slot-0" \
                    and desc["index"] + 1 == desc["total"]:
                first["first_image_ms"] = tick_ms
            elif desc["media_type"] == "video":
                if first["i2v_first_public_chunk_ms"] is None:
                    first["i2v_first_public_chunk_ms"] = tick_ms
                if desc["index"] + 1 == desc["total"]:
                    first["i2v_playable_ms"] = tick_ms
    taps = load(arm_dir / "native-taps.json")
    origin = arm["interactive_timings"]["capture_origin_monotonic_ns"]
    first["first_tts_submit_ms"] = None if not taps["tts"] else \
        (taps["tts"][0]["observed_monotonic_ns"] - origin) / 1e6
    first["i2v_native_submit_ms"] = None if not taps["video"] else \
        (taps["video"][0]["observed_monotonic_ns"] - origin) / 1e6
    lifecycle = arm["stage_lifecycle"]
    first["tts_ready_ms"] = None if lifecycle.get("tts_ready_monotonic_ns") is None else \
        (lifecycle["tts_ready_monotonic_ns"] - origin) / 1e6
    first["wan_ready_ms"] = None if lifecycle.get("wan_ready_monotonic_ns") is None else \
        (lifecycle["wan_ready_monotonic_ns"] - origin) / 1e6
    expected_measurements = {key: ms(value) for key, value in first.items()}
    expected_measurements["asr"] = "NOT_APPLICABLE_TEXT_INPUT"
    require(expected_measurements == expected["measurements"],
            f"timing mismatch: {name}: {expected_measurements} != {expected['measurements']}")
    thresholds = prereg["hard_latency_thresholds_ms"]
    hard = {}
    for metric, threshold in (("ttft_ms", "ttft_strictly_below"), ("ttfa_ms", "ttfa_strictly_below"),
                              ("first_image_ms", "first_image_strictly_below"),
                              ("i2v_playable_ms", "i2v_playable_strictly_below")):
        value = expected_measurements[metric]
        hard[metric] = {"value_ms": value, "strictly_below_ms": thresholds[threshold],
                        "result": "MET" if value is not None and value < thresholds[threshold] else "NOT_MET"}
    require(hard == expected["hard_thresholds"], f"hard threshold mismatch: {name}")

    artifacts = []
    grouped = {}
    slot_sequence = {event["slot_id"]: event["seq"] for event in slots}
    for event in public_events:
        if event["kind"] != "artifact":
            continue
        desc = event["artifact"]
        if desc.get("slot_id") is not None:
            require(desc["slot_id"] in slot_sequence and slot_sequence[desc["slot_id"]] < event["seq"],
                    f"artifact precedes its visual slot: {name}")
        grouped.setdefault(desc["artifact_id"], []).append(desc)
    for artifact_id, chunks in grouped.items():
        chunks.sort(key=lambda item: item["index"])
        require([item["index"] for item in chunks] == list(range(len(chunks)))
                and all(item["total"] == len(chunks) for item in chunks), f"artifact chunks invalid: {name}")
        joined = b"".join(base64.b64decode(item["data"], validate=True) for item in chunks)
        last = chunks[-1]
        require(sha(joined) == last["sha256"], f"joined artifact digest mismatch: {name}")
        bound = next(item for item in record["artifacts"] if item["artifact_id"] == artifact_id)
        require((arm_dir / "capture" / "artifacts" / bound["ref"]).read_bytes() == joined,
                f"CAS bytes mismatch: {name}")
        artifacts.append((bound, joined))
    images = [item for item in artifacts if item[0]["media_type"] == "image"]
    videos = [item for item in artifacts if item[0]["media_type"] == "video"]
    require((images[0][0] if images else None) == expected["source"], f"image projection mismatch: {name}")
    require((videos[0][0] if videos else None) == expected["video"], f"video projection mismatch: {name}")

    outputs = taps["tts_output"]
    logs = native_tts_logs(arm_dir / "tts.log")
    public_audio = [(bound, raw) for bound, raw in artifacts if bound["media_type"] == "audio"]
    legs = []
    public_cursor = 0
    joins = {item["request_id"]: item for item in arm["tts_legs"]}
    for ordinal, submit in enumerate(taps["tts"]):
        request_id = submit["request_id"]
        packed_submit = msgpack.unpackb((arm_dir / submit["packed_file"]).read_bytes(), raw=False)
        text = packed_submit["data"]["request"]["inputs"]["text"]
        require(sha(text.encode()) == submit["text_sha256"], f"TTS text hash mismatch: {name}:{request_id}")
        streams = [item for item in outputs if item.get("message_type") == "stream"
                   and item.get("request_id") == request_id]
        terminals = [item for item in outputs if item.get("message_type") == "complete"
                     and item.get("request_id") == request_id]
        native_pcm = bytearray()
        native_order = bool(streams)
        public_exact = True
        for index, stream in enumerate(streams):
            packed = msgpack.unpackb((arm_dir / stream["packed_file"]).read_bytes(), raw=False)
            chunk = packed["chunk"]["audio"]
            native_order = native_order and stream["stream_index"] == index \
                and sha(chunk) == stream["audio_sha256"]
            native_pcm.extend(chunk)
            if public_cursor >= len(public_audio):
                public_exact = False
                continue
            public, public_raw = public_audio[public_cursor]
            public_exact = public_exact and public.get("stream_index") == index and public_raw == chunk
            public_cursor += 1
        wav_exact = False
        terminal_success = False
        if len(terminals) == 1:
            terminal_success = terminals[0].get("success") is True
            packed = msgpack.unpackb((arm_dir / terminals[0]["packed_file"]).read_bytes(), raw=False)
            wav_raw = (packed.get("result") or {}).get("data")
            if terminal_success and isinstance(wav_raw, bytes):
                with wave.open(io.BytesIO(wav_raw), "rb") as wav:
                    wav_exact = wav.getnchannels() == 1 and wav.getsampwidth() == 2 \
                        and wav.getframerate() == 24000 and wav.readframes(wav.getnframes()) == bytes(native_pcm)
        log = logs.get(request_id, {"codec_frames": None, "natural_eos": False,
                                   "samples": None, "native_elapsed_ms": None})
        legs.append({"ordinal": ordinal, "request_id": request_id,
                     "text_sha256": submit["text_sha256"], "chunk_count": len(streams),
                     "pcm_bytes": len(native_pcm), "pcm_sha256": sha(bytes(native_pcm)),
                     "native_chunk_order_exact": native_order, "public_pcm_exact": public_exact,
                     "native_terminal_count": len(terminals), "native_terminal_success": terminal_success,
                     "terminal_wav_exact": wav_exact,
                     "producer_join_exact": joins.get(request_id, {}).get(
                         "exact_native_public_wav_join", False), **log})
    require(legs == expected["tts_legs"], f"per-leg PCM/EOS mismatch: {name}")
    aggregate = b"".join(raw for _, raw in public_audio)
    tts_text = "".join(item["text"] for item in taps["tts"])
    pcm = {"public_chunks": len(public_audio), "public_bytes": len(aggregate),
           "public_sha256": sha(aggregate), "unowned_public_chunks": len(public_audio) - public_cursor,
           "first_tts_text": taps["tts"][0]["text"] if taps["tts"] else None,
           "tts_text_matches_clean": spoken_text_projection(tts_text) ==
                                     spoken_text_projection(clean),
           "native_tts_tag_leaks": len(re.findall(
               r"\[\[\s*(?:image|video|visual)\s*:|source=slot-\d+", tts_text, flags=re.I)),
           "native_tts_markdown_presentation_leaks": sum(
               native_speech_has_presentation_tokens(item["text"]) for item in taps["tts"])}
    require(pcm == expected["pcm"], f"aggregate PCM mismatch: {name}")
    if taps["video"]:
        require(images, f"I2V source missing: {name}")
        submit = taps["video"][0]
        packed = msgpack.unpackb((arm_dir / submit["packed_file"]).read_bytes(), raw=False)
        inputs = packed["data"]["request"]["inputs"]
        require(inputs["image"] == images[0][1] and inputs["anchor_explicit"] is True
                and inputs["anchor_slot_index"] == 0 and inputs["anchor_sha256"] == sha(images[0][1]),
                f"I2V exact anchor mismatch: {name}")
    finish = vlm_finish(arm_dir / "vlm.log")
    require(finish == expected["vlm_finish"], f"VLM finish mismatch: {name}")
    video_stream = None
    if videos:
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=codec_name,width,height,nb_frames,r_frame_rate,pix_fmt", "-of", "json",
            str(arm_dir / "capture" / "artifacts" / videos[0][0]["ref"])],
            check=True, capture_output=True, text=True, timeout=30)
        video_stream = json.loads(probe.stdout)["streams"][0]
        require(video_stream["codec_name"] == "h264" and video_stream["pix_fmt"] == "yuv420p"
                and (video_stream["width"], video_stream["height"]) == (704, 480)
                and int(video_stream["nb_frames"]) == 29 and video_stream["r_frame_rate"] == "16/1",
                f"playable video contract mismatch: {name}")
    require(video_stream == expected["video_stream"], f"video stream projection mismatch: {name}")
    for key, filename in (("server_log_sha256", "server.log"), ("vlm_log_sha256", "vlm.log"),
                          ("tts_log_sha256", "tts.log"), ("video_log_sha256", "video.log")):
        value = file_sha(arm_dir / filename) if (arm_dir / filename).exists() else None
        require(value == expected["raw"][key], f"raw log hash mismatch: {name}:{filename}")
    require(expected["event_counts"] == {
        "public_events": len(public_events), "public_artifacts": len(artifacts),
        "audio_artifacts": len(public_audio)}, f"event/artifact counts mismatch: {name}")
    failures = list(arm["failures"])
    require(arm["result"] == ("CAPTURED_WITH_FAILURES" if failures else "CAPTURED"),
            f"capture result/failure projection mismatch: {name}")
    def fail(condition, label):
        if not condition:
            failures.append(label)
    fail(terminal_state == "completed", "PUBLIC_TERMINAL_NOT_COMPLETE")
    fail(finish["finish_reason"] == "stop", "VLM_TRUNCATED_OR_FINISH_REASON_MISSING")
    for key in ("opening_before_image", "opening_is_concise", "opening_has_no_heading",
                "image_slot_present", "video_slot_present", "video_explicit_anchor",
                "minimum_clean_text_bytes_met", "minimum_opening_paragraphs_met",
                "minimum_background_met", "minimum_headings_met", "minimum_terms_met"):
        fail(content[key], "CONTENT_" + key.upper())
    fail(content["tag_leaks"] == 0, "PUBLIC_TAG_OR_DIRECTIVE_LEAK")
    fail(bool(images), "FIRST_IMAGE_MISSING")
    fail(bool(videos), "I2V_MEDIA_MISSING")
    fail(video_stream is not None, "I2V_NOT_PLAYABLE")
    fail(bool(legs), "NO_NATIVE_TTS_LEGS")
    for leg in legs:
        for key in ("natural_eos", "native_chunk_order_exact", "public_pcm_exact",
                    "native_terminal_success", "terminal_wav_exact", "producer_join_exact"):
            fail(leg[key], f"TTS_{key.upper()}:{leg['request_id']}")
        fail(leg["native_terminal_count"] == 1, f"TTS_TERMINAL_COUNT:{leg['request_id']}")
    fail(pcm["unowned_public_chunks"] == 0, "UNOWNED_PUBLIC_PCM")
    fail(pcm["tts_text_matches_clean"], "TTS_TEXT_DIFFERS_FROM_CLEAN_PROSE")
    fail(pcm["native_tts_tag_leaks"] == 0, "NATIVE_TTS_TAG_OR_DIRECTIVE_LEAK")
    fail(pcm["native_tts_markdown_presentation_leaks"] == 0,
         "NATIVE_TTS_MARKDOWN_PRESENTATION_LEAK")
    fail(pcm["first_tts_text"] is not None and
         pcm["first_tts_text"].strip() == content["opening"], "FIRST_TTS_IS_NOT_CONCISE_OPENING")
    fail(arm["health_final"] is not None and arm["health_final"].get("in_flight") == 0,
         "FINAL_IN_FLIGHT_NONZERO")
    require(sorted(set(failures)) == expected["failures"], f"failure list mismatch: {name}")
    return {"events": len(public_events), "artifacts": len(artifacts), "tts_legs": len(legs),
            "pcm_bytes": len(aggregate), "clean_text_bytes": len(clean.encode())}


def verify_decisions(scorecard, prereg, candidate):
    arms = scorecard["arms"]
    pairs = []
    for case in ("chinese", "english"):
        for repetition in (1, 2):
            before = arms[f"{case}-r{repetition}-before"]
            after = arms[f"{case}-r{repetition}-after"]
            timing = {}
            for metric in ("ttft_ms", "first_tts_submit_ms", "ttfa_ms", "image_slot_ms",
                           "first_image_ms", "video_slot_ms", "i2v_native_submit_ms", "i2v_playable_ms"):
                old, new = before["measurements"][metric], after["measurements"][metric]
                timing[metric] = {"before_ms": old, "after_ms": new,
                                  "after_minus_before_ms": None if old is None or new is None else round(new-old, 3),
                                  "improved": new is not None and (old is None or new < old),
                                  "missing_before": old is None, "missing_after": new is None}
            pairs.append({"before": f"{case}-r{repetition}-before", "after": f"{case}-r{repetition}-after",
                          "timing": timing, "after_failures": after["failures"],
                          "before_failures": before["failures"]})
    require(pairs == scorecard["pairs"], "paired score mismatch")
    aggregates = {}
    for case in ("chinese", "english"):
        for arm in ("before", "after"):
            selected = [v for v in arms.values() if v["case"] == case and v["arm"] == arm]
            aggregates[f"{case}-{arm}"] = {metric: median([v["measurements"][metric] for v in selected])
                for metric in ("ttft_ms", "first_tts_submit_ms", "ttfa_ms", "first_image_ms",
                               "video_slot_ms", "i2v_native_submit_ms", "i2v_playable_ms")}
    require(aggregates == scorecard["aggregates"], "aggregate timing mismatch")
    ttfa_before, ttfa_after = aggregates["chinese-before"]["ttfa_ms"], aggregates["chinese-after"]["ttfa_ms"]
    image_before = aggregates["chinese-before"]["first_image_ms"]
    image_after = aggregates["chinese-after"]["first_image_ms"]
    ttfa_gain = None if ttfa_before in (None, 0) or ttfa_after is None else (ttfa_before-ttfa_after)/ttfa_before
    image_gain = None if image_before in (None, 0) or image_after is None else (image_before-image_after)/image_before
    integrity = all(not arm["failures"] for arm in arms.values() if arm["arm"] == "after")
    repeated = all(pair["timing"]["ttfa_ms"]["improved"] and pair["timing"]["first_image_ms"]["improved"]
                   for pair in pairs)
    rule = candidate["promotion_rule"]
    latency = ttfa_gain is not None and ttfa_gain >= rule["minimum_chinese_median_ttfa_reduction_fraction"] \
        and image_gain is not None and image_gain >= rule["minimum_chinese_median_first_image_reduction_fraction"]
    improvement = {"chinese_median_ttfa_reduction_fraction": ttfa_gain,
                   "chinese_median_first_image_reduction_fraction": image_gain,
                   "every_pair_ttfa_and_first_image_improved": repeated,
                   "after_text_media_lifecycle_integrity_pass": integrity,
                   "required_controls_pass": scorecard["controls"].get("all_required_pass") is True}
    require(improvement == scorecard["paired_improvement"], "promotion inputs mismatch")
    decision = "PROMOTED_ENGINEERING_GAIN" if integrity and repeated and latency and improvement[
        "required_controls_pass"] else "NOT_PROMOTED"
    require(decision == scorecard["promotion_decision"], "promotion decision mismatch")
    not_met = sorted({f"{key}:{metric}" for key, arm in arms.items()
                      for metric, result in arm["hard_thresholds"].items() if result["result"] == "NOT_MET"})
    require(not_met == scorecard["hard_threshold_not_met"], "NOT_MET list mismatch")
    require(sum(len(arm["failures"]) for arm in arms.values()) == scorecard["all_failure_count"],
            "failure count mismatch")


def write_manifest(bundle):
    rows = []
    for path in sorted(p for p in bundle.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        rows.append(f"{file_sha(path)}  {path.relative_to(bundle).as_posix()}\n")
    (bundle / "SHA256SUMS").write_text("".join(rows))
    return len(rows)


def verify_manifest(bundle):
    bound = {}
    for line in (bundle / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        bound[name] = digest
    files = {p.relative_to(bundle).as_posix(): p for p in bundle.rglob("*")
             if p.is_file() and p.name != "SHA256SUMS"}
    require(set(bound) == set(files), "top-level manifest file set mismatch")
    for name, path in files.items():
        require(file_sha(path) == bound[name], f"top-level manifest mismatch: {name}")
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.write_manifest:
        count = write_manifest(args.bundle)
        print(json.dumps({"manifest_files": count,
                          "manifest_sha256": file_sha(args.bundle / "SHA256SUMS")}, sort_keys=True))
        return 0
    scorecard = load(args.bundle / "scorecard.json")
    prereg = load(args.bundle / "preregistration.json")
    candidate = load(args.bundle / "candidate.json")
    require(scorecard["format"] == "umminfer-ns01-streaming-opening-scorecard-v1", "scorecard format")
    require(candidate == scorecard["candidate"], "candidate binding mismatch")
    require(file_sha(args.bundle / "preregistration.json") == scorecard["preregistration_sha256"],
            "preregistration binding mismatch")
    counts = {}
    cache = {"models": None, "binaries": {}}
    for name, expected in scorecard["arms"].items():
        counts[name] = verify_arm(args.bundle, name, expected, prereg, candidate, cache)
    verify_decisions(scorecard, prereg, candidate)
    for attempt in scorecard.get("retained_infrastructure_attempts", []):
        directory = args.bundle / "negative" / attempt["name"]
        files = {path.relative_to(directory).as_posix(): file_sha(path)
                 for path in sorted(directory.rglob("*")) if path.is_file()
                 and "run" not in path.relative_to(directory).parts}
        arm_path = directory / "arm.json"
        failures = load(arm_path).get("failures", []) if arm_path.exists() else [
            "pre-request stage startup failure; see retained image/vlm logs"]
        require(attempt == {"name": directory.name, "file_count": len(files),
                            "tree_sha256": sha(canonical(files)), "failures": failures,
                            "accepted_measurement": False,
                            "reason": "infrastructure-only; no accepted public measurement or candidate tuning"},
                f"infrastructure attempt projection mismatch: {directory.name}")
    entries = scorecard["controls"].get("entries", {})
    require(entries and scorecard["controls"].get("all_required_pass") ==
            all(control.get("pass") is True for control in entries.values()),
            "aggregate control outcome mismatch")
    for name, control in entries.items():
        path = args.bundle / control["path"]
        require(file_sha(path) == control["sha256"], f"control raw hash mismatch: {name}")
        require(isinstance(control.get("pass"), bool),
                f"control lacks an explicit pass/fail outcome: {name}")
    file_count = verify_manifest(args.bundle)
    report = {"format": "umminfer-ns01-streaming-opening-independent-verification-v1",
              "result": "PASS", "manifest_files": file_count,
              "manifest_sha256": file_sha(args.bundle / "SHA256SUMS"),
              "scorecard_sha256": file_sha(args.bundle / "scorecard.json"),
              "arms": counts, "promotion_decision": scorecard["promotion_decision"],
              "all_required_controls_pass": scorecard["controls"].get("all_required_pass"),
              "human_perceptual_evaluation": "NOT_RUN"}
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        require(not args.out.exists(), f"verification output exists: {args.out}")
        args.out.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    raise SystemExit(main())
