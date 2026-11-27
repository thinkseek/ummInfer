#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a compact NS-01 TTS lifecycle scorecard from retained raw evidence.

All paths are caller-selected. The output directory must be new. Raw evidence is
hard-linked into the external bundle when possible and covered by one complete
self-excluded SHA256SUMS. No media or per-event payload is written to Git.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import wave

FORMAT = "umminfer-ns01-ns04-tts-lifecycle-scorecard-v1"
TAGS = ("[[image:", "[[video:", "[[visual:")


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
    size = 0
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
            size += len(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hardlink_tree(source, destination):
    def link(src, dst):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    shutil.copytree(source, destination, copy_function=link)


def tree_identity(root):
    files = {path.relative_to(root).as_posix(): file_sha(path)
             for path in sorted(root.rglob("*")) if path.is_file()}
    return {"file_count": len(files), "tree_sha256": sha(canonical(files))}


def parse_sse(raw):
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
        require(data, "non-data frame in narrative capture")
        frames.append((frame_raw, b"\n".join(data)))
        cursor = end + 2
    return frames


def speech_projection(text):
    text = re.sub(r"(?m)^ {0,3}(?:#+[ \t]+|>[ \t]?|[*+-][ \t]+)", "", text)
    text = re.sub(r"(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", "", text)
    for marker in ("**", "__", "~~", "`"):
        text = text.replace(marker, "")
    return "".join(text.split())


def native_records(root):
    import msgpack
    submits = {}
    outputs = {}
    submit_pattern = re.compile(r"tts-(\d+)-[0-9a-f]{64}\.msgpack$")
    output_pattern = re.compile(r"tts-output-(\d+)-[0-9a-f]{64}\.msgpack$")
    for path in sorted(root.glob("tts-*.msgpack")):
        match = submit_pattern.match(path.name)
        if not match:
            continue
        message = msgpack.unpackb(path.read_bytes(), raw=False)
        if message.get("type") != "submit":
            continue
        request_id = message.get("request_id")
        text = (((message.get("data") or {}).get("request") or {})
                .get("inputs") or {}).get("text")
        require(isinstance(request_id, str) and isinstance(text, str),
                f"invalid native TTS submit: {path.name}")
        require(request_id not in submits, "duplicate native submit ordinal")
        submits[request_id] = {"ordinal": int(match.group(1)), "text": text,
                               "packed": file_sha(path), "mtime_ns": path.stat().st_mtime_ns}
    for path in sorted(root.glob("tts-output-*.msgpack")):
        match = output_pattern.match(path.name)
        if not match:
            continue
        message = msgpack.unpackb(path.read_bytes(), raw=False)
        if message.get("type") == "stage_ready":
            continue
        request_id = message.get("request_id")
        require(isinstance(request_id, str), f"native output lacks request id: {path.name}")
        item = {"ordinal": int(match.group(1)), "message": message,
                "packed": file_sha(path), "mtime_ns": path.stat().st_mtime_ns}
        outputs.setdefault(request_id, []).append(item)
    for rows in outputs.values():
        rows.sort(key=lambda item: item["ordinal"])
    return dict(sorted(submits.items(), key=lambda item: item[1]["ordinal"])), outputs


def native_log(root):
    path = root / "tts.log"
    text = path.read_text(encoding="utf-8", errors="replace")
    complete = {}
    aborted = {}
    for line in text.splitlines():
        match = re.search(r"tts request request_id=([^\s]+) (.*)$", line)
        if match:
            fields = dict(re.findall(r"([a-z0-9_]+)=([^\s]+)", match.group(2)))
            complete[match.group(1)] = fields
        match = re.search(r"tts request aborted request_id=([^\s]+) (.*)$", line)
        if match:
            aborted[match.group(1)] = dict(
                re.findall(r"([a-z0-9_]+)=([^\s]+)", match.group(2)))
    return complete, aborted, file_sha(path), text


def pcm_from_wav(raw):
    import io
    with wave.open(io.BytesIO(raw), "rb") as wav:
        require(wav.getnchannels() == 1 and wav.getsampwidth() == 2 and
                wav.getframerate() == 24000, "unexpected native terminal WAV format")
        return wav.readframes(wav.getnframes())


def active_highwater(submits, outputs):
    events = []
    for request_id, submit in submits.items():
        events.append((submit["mtime_ns"], 0, request_id, 1))
        for output in outputs.get(request_id, []):
            if output["message"].get("type") == "complete":
                events.append((output["mtime_ns"], 1, request_id, -1))
    active = 0
    highwater = 0
    valid = True
    for _mtime, _tie, _request_id, delta in sorted(events):
        active += delta
        if active < 0:
            valid = False
            active = 0
        highwater = max(highwater, active)
    return {"active_highwater": highwater, "final_unsettled": active,
            "ordered_event_join_valid": valid,
            "clock": "filesystem_mtime_ns_diagnostic"}


def summarize_arm(root, label, prereg):
    root = Path(root)
    request_path = root / "request.json"
    request = load(request_path)
    expected = prereg["golden_case"]
    require(request["messages"][0]["content"] == expected["user_text"] and
            request["max_tokens"] == expected["max_tokens"] and
            request["temperature"] == expected["temperature"] and
            request["seed"] == expected["seed"] and
            request["narrative"] == {"enabled": True, "visual_mode": "auto",
                                     "max_slots": 2, "allow_text_only": False,
                                     "continuity_mode": "i2v_chain"} and
            "teacache_thresh" not in request["video"],
            f"{label}: request differs from frozen preregistration")
    trajectory_path = root / "capture" / "trajectories.jsonl"
    lines = trajectory_path.read_text(encoding="utf-8").splitlines()
    require(len(lines) == 1, f"{label}: expected one public trajectory")
    trajectory = json.loads(lines[0])
    replay = trajectory["sse_replay"]
    raw_ref = Path(replay["raw_ref"])
    require(not raw_ref.is_absolute() and ".." not in raw_ref.parts,
            "unsafe raw SSE reference")
    raw_path = root / "capture" / "artifacts" / raw_ref
    raw = raw_path.read_bytes()
    require(sha(raw) == replay["raw_sha256"] and len(raw) == replay["raw_bytes"],
            f"{label}: raw SSE identity mismatch")
    frames = parse_sse(raw)
    tick_by_frame = {event["source_event_index"]: event["source_event_tick"]
                     for event in trajectory["events"]}
    events = []
    done_count = 0
    for index, (_frame_raw, payload) in enumerate(frames):
        if payload == b"[DONE]":
            done_count += 1
            continue
        frame = json.loads(payload)
        if frame.get("unified") is not None:
            events.append((index, frame["unified"], tick_by_frame.get(index)))
    clean = "".join(event.get("text", "") for _index, event, _tick in events
                    if event.get("kind") == "vlm_delta")
    require(clean == trajectory["clean_text"], f"{label}: clean text differs from trajectory")
    public_audio = []
    terminals = []
    visual_slots = []
    first_text = first_audio = first_image = video_complete = terminal_tick = None
    public_image_count = public_video_count = 0
    seen_media = set()
    for frame_index, event, tick in events:
        kind = event.get("kind")
        if kind == "vlm_delta" and event.get("text") and first_text is None:
            first_text = tick
        if kind == "visual_slot":
            visual_slots.append({"slot_id": event.get("slot_id"),
                                 "stage": event.get("stage")})
        if kind == "artifact":
            descriptor = event["artifact"]
            media = descriptor.get("media_type")
            final_chunk = descriptor.get("index") + 1 == descriptor.get("total")
            if final_chunk and descriptor.get("artifact_id") not in seen_media:
                seen_media.add(descriptor.get("artifact_id"))
                if media == "image":
                    public_image_count += 1
                    if descriptor.get("slot_id") == "slot-0" and first_image is None:
                        first_image = tick
                elif media == "video":
                    public_video_count += 1
                    video_complete = tick
            if media == "audio":
                pcm = base64.b64decode(descriptor["data"], validate=True)
                public_audio.append(pcm)
                if first_audio is None:
                    first_audio = tick
        if kind in ("complete", "cancelled", "error"):
            terminals.append(event)
            terminal_tick = tick
    require(done_count == 1 and len(terminals) == 1,
            f"{label}: public response must have one terminal and one DONE")
    submits, outputs = native_records(root)
    completed_logs, aborted_logs, log_identity, log_text = native_log(root)
    legs = []
    public_cursor = 0
    for request_id, submit in submits.items():
        native_chunks = []
        terminal_messages = []
        packed_outputs = []
        for output in outputs.get(request_id, []):
            packed_outputs.append(output["packed"])
            message = output["message"]
            if message.get("type") == "stream":
                chunk = message.get("chunk") or {}
                audio = chunk.get("audio")
                require(isinstance(audio, bytes) and audio, "empty native PCM chunk")
                native_chunks.append((chunk.get("index"), audio))
            elif message.get("type") == "complete":
                terminal_messages.append(message)
        native_chunks.sort(key=lambda item: item[0])
        require([index for index, _pcm in native_chunks] ==
                list(range(len(native_chunks))), f"{request_id}: native stream index gap")
        native_pcm = b"".join(pcm for _index, pcm in native_chunks)
        public_pcm = b""
        public_leg_chunks = 0
        native_public_exact = True
        for _index, pcm in native_chunks:
            if public_cursor >= len(public_audio) or public_audio[public_cursor] != pcm:
                native_public_exact = False
                break
            public_pcm += pcm
            public_cursor += 1
            public_leg_chunks += 1
        terminal_pcm = None
        terminal_wav_sha = None
        native_success = False
        for message in terminal_messages:
            result = message.get("result") or {}
            wav = result.get("data")
            if isinstance(wav, bytes) and wav:
                terminal_wav_sha = sha(wav)
                terminal_pcm = pcm_from_wav(wav)
            native_success = native_success or message.get("success") is True
        log_fields = completed_logs.get(request_id, {})
        natural_eos = None
        generated_frames = None
        if "natural_eos" in log_fields:
            natural_eos = log_fields["natural_eos"] == "1"
        if "codec_frames" in log_fields:
            generated_frames = int(log_fields["codec_frames"])
        state = ("completed" if native_success else "aborted" if request_id in aborted_logs
                 else "nonterminal")
        legs.append({
            "child_id": request_id, "submit_ordinal": submit["ordinal"],
            "text_sha256": sha(submit["text"].encode()),
            "text_bytes": len(submit["text"].encode()),
            "text": submit["text"],
            "native_stream_chunks": len(native_chunks),
            "native_pcm_bytes": len(native_pcm), "native_pcm_sha256": sha(native_pcm),
            "native_terminal_count": len(terminal_messages),
            "native_terminal_success": native_success,
            "native_stream_terminal_pcm_exact": terminal_pcm == native_pcm if terminal_pcm is not None else None,
            "terminal_wav_sha256": terminal_wav_sha,
            "public_chunks": public_leg_chunks,
            "public_pcm_bytes": len(public_pcm), "public_pcm_sha256": sha(public_pcm),
            "native_public_pcm_exact": native_public_exact,
            "natural_eos": natural_eos, "generated_frames": generated_frames,
            "state": state, "native_output_file_count": len(packed_outputs),
            "submit_file": submit["packed"],
        })
    all_public = b"".join(public_audio)
    tts_text = "".join(leg["text"] for leg in legs)
    presentation_leaks = sum(
        bool(re.search(r"(?:\*\*|__|~~|`)", leg["text"]) or
             re.search(r"(?m)^ {0,3}#+[ \t]+", leg["text"]) or
             re.search(r"(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", leg["text"]))
        for leg in legs)
    for leg in legs:
        del leg["text"]
    clean_bytes = clean.encode()
    before_heading = re.split(r"(?m)^#", clean, maxsplit=1)[0]
    opening_paragraphs = [part for part in re.split(r"\n\s*\n", before_heading)
                          if part.strip()]
    hard = {}
    timings = {"ttft_ms": None if first_text is None else first_text / 1e6,
               "ttfa_ms": None if first_audio is None else first_audio / 1e6,
               "first_image_ms": None if first_image is None else first_image / 1e6,
               "i2v_complete_ms": None if video_complete is None else video_complete / 1e6,
               "total_to_terminal_ms": None if terminal_tick is None else terminal_tick / 1e6}
    for key, limit in (("ttft_ms", 800.0), ("ttfa_ms", 800.0),
                       ("first_image_ms", 2000.0), ("i2v_complete_ms", 5000.0)):
        hard[key] = "NOT_RUN" if timings[key] is None else (
            "PASS" if timings[key] < limit else "NOT_MET")
    result = {
        "label": label, "request_sha256": file_sha(request_path)["sha256"],
        "trajectory_sha256": file_sha(trajectory_path)["sha256"],
        "raw_response_sha256": sha(raw), "raw_response_bytes": len(raw),
        "public_event_count": len(events), "public_done_count": done_count,
        "public_terminal_kind": terminals[0]["kind"],
        "public_terminal_count": 1,
        "trajectory_terminal_state": trajectory["terminal_state"],
        "clean_text_sha256": sha(clean_bytes), "clean_text_bytes": len(clean_bytes),
        "heading_lines": sum(line.startswith("#") for line in clean.splitlines()),
        "opening_paragraphs_before_first_heading": len(opening_paragraphs),
        "opening_chinese_codepoints": sum("一" <= c <= "鿿" for c in before_heading),
        "mixed_terms_present": [term for term in
            ("accretion disk", "gravitational lensing", "event horizon", "Eddington limit")
            if term in clean],
        "tag_leak_count": sum(clean.count(tag) + tts_text.count(tag) for tag in TAGS),
        "native_markdown_presentation_leak_count": presentation_leaks,
        "speech_text_projection_exact": speech_projection(tts_text) == speech_projection(clean),
        "tts_leg_count": len(legs),
        "native_terminal_success_count": sum(leg["native_terminal_success"] for leg in legs),
        "native_nonterminal_count": sum(leg["state"] == "nonterminal" for leg in legs),
        "native_aborted_count": sum(leg["state"] == "aborted" for leg in legs),
        "natural_eos_count": sum(leg["natural_eos"] is True for leg in legs),
        "natural_eos_unknown_count": sum(leg["natural_eos"] is None for leg in legs),
        "native_stream_terminal_exact_count": sum(
            leg["native_stream_terminal_pcm_exact"] is True for leg in legs),
        "native_public_exact_leg_count": sum(leg["native_public_pcm_exact"] for leg in legs),
        "public_pcm_chunks": len(public_audio), "public_pcm_bytes": len(all_public),
        "public_pcm_sha256": sha(all_public),
        "unmatched_public_pcm_chunks": len(public_audio) - public_cursor,
        "tts_queue": active_highwater(submits, outputs),
        "timings": timings, "hard_latency": hard,
        "visual_slots": visual_slots,
        "public_image_count": public_image_count,
        "public_video_count": public_video_count,
        "native_tts_log": log_identity,
        "native_log_tail_sha256": sha(log_text[-8192:].encode()),
        "legs": legs,
    }
    if (root / "identity.json").is_file():
        result["identity"] = load(root / "identity.json")
        result["identity_sha256"] = file_sha(root / "identity.json")["sha256"]
    result["all_candidate_correctness_gates"] = bool(
        legs and result["native_terminal_success_count"] == len(legs) and
        result["native_nonterminal_count"] == 0 and
        result["natural_eos_count"] == len(legs) and
        result["native_stream_terminal_exact_count"] == len(legs) and
        result["native_public_exact_leg_count"] == len(legs) and
        result["unmatched_public_pcm_chunks"] == 0 and
        result["public_terminal_kind"] == "complete" and
        result["trajectory_terminal_state"] == "completed" and
        result["tag_leak_count"] == 0 and
        result["native_markdown_presentation_leak_count"] == 0 and
        result["speech_text_projection_exact"])
    result["ideal_single_stream_correctness"] = bool(
        result["all_candidate_correctness_gates"] and
        [(item["slot_id"], item["stage"]) for item in visual_slots] ==
            [("slot-0", "image"), ("slot-1", "video")] and
        public_image_count == 1 and public_video_count == 1 and
        result["clean_text_bytes"] >= expected["minimum_clean_text_bytes"] and
        result["opening_paragraphs_before_first_heading"] >=
            expected["minimum_opening_paragraphs_before_first_heading"] and
        result["opening_chinese_codepoints"] >=
            expected["minimum_opening_chinese_codepoints_before_first_heading"] and
        result["heading_lines"] >= expected["required_markdown_heading_lines"] and
        len(result["mixed_terms_present"]) >= expected["required_mixed_terms_any"])
    return result


def summarize_ns04(root):
    root = Path(root)
    evidence_path = root / "full-duplex-barge-in-evidence.json"
    evidence = load(evidence_path)
    raw_path = root / "raw-events.jsonl"
    raw_identity = file_sha(raw_path)
    records = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    previous = -1
    rx = []
    markers = []
    for record in records:
        raw = record["raw_message"].encode()
        require(sha(raw) == record["raw_sha256"] and
                record["observed_monotonic_ns"] >= previous,
                "NS04 raw record hash/order mismatch")
        previous = record["observed_monotonic_ns"]
        event = json.loads(record["raw_message"])
        item = (record["observed_monotonic_ns"], event)
        if record["direction"] == "rx":
            rx.append(item)
        elif record["direction"] == "marker":
            markers.append(item)
    require(raw_identity == evidence["raw_events"] and
            len(records) == evidence["raw_record_count"] and
            len(rx) == evidence["raw_event_count"], "NS04 raw/evidence count mismatch")
    first_id = evidence["first_response_id"]
    second_id = evidence["second_response_id"]
    first_start = next(at for at, event in markers
                       if event.get("type") == "client_barge_start" and
                       event.get("response_id") == first_id)
    second_start = next(at for at, event in markers
                        if event.get("type") == "client_barge_start" and
                        event.get("response_id") == second_id)
    first_cancel = next((at, event) for at, event in rx
                        if event.get("type") == "response.cancelled" and
                        event.get("response_id") == first_id)
    second_cancel = next((at, event) for at, event in rx
                         if event.get("type") == "response.cancelled" and
                         event.get("response_id") == second_id)
    silence = next(at for at, event in markers
                   if event.get("type") == "digital_silence" and
                   event.get("response_id") == first_id)
    transcripts = [at for at, event in rx if event.get("type") == "transcription.completed"]
    second_audio = next(at for at, event in rx
                        if event.get("type") == "audio.delta" and
                        event.get("response_id") == second_id)
    require(len(transcripts) >= 2, "NS04 lacks post-barge ASR final")
    first_audio = [event for at, event in rx
                   if at <= first_cancel[0] and event.get("type") == "audio.delta" and
                   event.get("response_id") == first_id]
    played_count = evidence["played_audio_events"]
    require(played_count <= len(first_audio), "NS04 played count exceeds raw audio")
    played = b"".join(base64.b64decode(event["audio"], validate=True)
                      for event in first_audio[:played_count])
    pending = first_audio[played_count:]
    pending_bytes = sum(len(base64.b64decode(event["audio"], validate=True))
                        for event in pending)
    require(sha(played) == evidence["played_pcm_sha256"] and
            len(played) == evidence["played_pcm_bytes"] and
            len(pending) == evidence["cancel_pending_audio_events"] and
            pending_bytes == evidence["cancel_pending_pcm_bytes"] and
            first_cancel[1].get("text_cursor") == evidence["acknowledged_cursor"],
            "NS04 played/pending/ACK mismatch")
    stale_types = {"audio.delta", "audio.done", "response.text.delta",
                   "response.playback_ack", "response.completed", "response.failed"}
    stale = sum(at > cancel[0] and event.get("response_id") == response_id and
                event.get("type") in stale_types
                for at, event in rx for cancel, response_id in
                ((first_cancel, first_id), (second_cancel, second_id)))
    timings = {
        "first_interrupt_ms": (first_cancel[0] - first_start) / 1e6,
        "cancel_to_digital_silence_ms": (silence - first_cancel[0]) / 1e6,
        "post_barge_ttfa_from_asr_final_ms": (second_audio - transcripts[1]) / 1e6,
        "barge_to_new_audio_ms": (second_audio - first_start) / 1e6,
        "second_interrupt_ms": (second_cancel[0] - second_start) / 1e6,
    }
    limits = {"first_interrupt_ms": 300.0, "cancel_to_digital_silence_ms": 300.0,
              "post_barge_ttfa_from_asr_final_ms": 800.0,
              "barge_to_new_audio_ms": 800.0, "second_interrupt_ms": 300.0}
    thresholds = {key: "PASS" if value < limits[key] else "NOT_MET"
                  for key, value in timings.items()}
    engineering = bool(evidence.get("status") == "PASS" and not evidence.get("error") and
                       stale == 0 and evidence["final_health"].get("in_flight") == 0 and
                       pending_bytes > 0 and
                       evidence["gpu_postflight"][str(evidence["gpus"]["asr"])]["memory_used_mib"] <= 64 and
                       evidence["gpu_postflight"][str(evidence["gpus"]["asr"])]["utilization_percent"] == 0 and
                       not evidence["gpu_postflight"][str(evidence["gpus"]["asr"])]["compute_processes"])
    return {"status": evidence["status"], "evidence_sha256": file_sha(evidence_path)["sha256"],
            "raw_events": raw_identity, "raw_records": len(records),
            "first_transcript": evidence["first_transcript"],
            "post_barge_transcript": evidence["post_barge_in_transcript"],
            "acknowledged_cursor": evidence["acknowledged_cursor"],
            "played_pcm_bytes": len(played), "played_pcm_sha256": sha(played),
            "pending_pcm_bytes": pending_bytes, "pending_audio_events": len(pending),
            "stale_output_events": stale, "final_in_flight": evidence["final_health"]["in_flight"],
            "timings_ms": timings, "hard_thresholds": thresholds,
            "engineering_pass": engineering,
            "physical_microphone_speaker": "NOT_RUN", "perceptual_quality": "NOT_RUN"}


def summarize_public_pcm(root):
    root = Path(root)
    first = root / "verification-r1.json"
    second = root / "verification-r2.json"
    require(first.read_bytes() == second.read_bytes(), "public PCM fresh reports differ")
    report = load(first)
    manifest = load(root / "capture-manifest.json")
    delayed = report["realtime"]["delayed_read"]
    counts = report["counts"]
    require(report["status"] == "PASS" and manifest["physical_gpu"] == 7 and
            counts["realtime_responses"] == 29 and delayed["paused_readers"] == 29 and
            delayed["native_audio_events_during_pause"] == 4096 and
            delayed["public_audio_events_during_pause"] == 0 and
            delayed["all_pcm_and_cursors_drained_exact"] and
            report["cancel_retry"]["retry_matches_fresh_repeat"] and
            report["cleanup"]["final_in_flight"] == 0 and
            report["cleanup"]["owned_processes_exited"], "public PCM gate is not complete")
    final_acks = sum(row["final_playback_ack"].get("final") is True
                     for row in report["realtime"]["responses"])
    require(final_acks == 29, "public PCM final ACK count mismatch")
    return {"status": "PASS", "verification_sha256": file_sha(first)["sha256"],
            "capture_manifest_sha256": file_sha(root / "capture-manifest.json")["sha256"],
            "nested_manifest_sha256": file_sha(root / "SHA256SUMS")["sha256"],
            "source_commit": report["source_commit"], "counts": counts,
            "final_playback_ack_count": final_acks,
            "delayed_read": {"paused_readers": 29, "native_audio_events_during_pause": 4096,
                             "public_audio_events_during_pause": 0,
                             "bounded_lossless_queue_boundary": 4096,
                             "pending_native_minus_public":
                                 delayed["release"]["pending_native_minus_public"],
                             "all_pcm_and_cursors_drained_exact": True},
            "cancel_retry": report["cancel_retry"], "cleanup": report["cleanup"],
            "physical_playback": "NOT_RUN", "perceptual_quality": "NOT_RUN"}


def summarize_network(root):
    root = Path(root)
    report_path = root / "independent-verification.json"
    run_path = root / "run.json"
    report = load(report_path)
    run = load(run_path)
    require(report["status"] == "PASS" and report["sessions"] == 6 and
            report["rounds"] == 3 and report["connections"] == 15 and
            report["responses"] == 41 and report["verified_next_turn_prompts"] == 30 and
            report["terminal_callbacks"] == 32 and report["stale_events"] == 0 and
            report["cross_session_leaks"] == 0 and report["final_in_flight"] == 0 and
            run["cleanup"]["selected_gpus_idle"], "network gate is incomplete")
    return {"status": "PASS", "verification_sha256": file_sha(report_path)["sha256"],
            "run_sha256": file_sha(run_path)["sha256"],
            "nested_manifest_sha256": file_sha(root / "SHA256SUMS")["sha256"],
            "source_commit": run["source_commit"], "sessions": 6, "rounds": 3,
            "connections": 15, "connection_scoped_response_identities": 41,
            "verified_next_turn_prompts": 30, "terminal_callbacks": 32,
            "stale_events": 0, "cross_session_leaks": 0, "final_in_flight": 0,
            "physical_microphone_speaker": "NOT_COVERED", "perceptual_quality": "NOT_COVERED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--ns04-parent", type=Path)
    parser.add_argument("--ns04-candidate", type=Path)
    parser.add_argument("--ns04-retained", action="append", type=Path, default=[])
    parser.add_argument("--public-pcm", type=Path)
    parser.add_argument("--network", type=Path)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--log", action="append", default=[],
                        help="NAME=PATH deterministic build/red/green evidence")
    parser.add_argument("--negative", action="append", default=[],
                        help="NAME=PATH raw incomplete/negative evidence directory")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out_dir.exists(), "scorecard output must be a new directory")
    args.out_dir.mkdir(parents=True)
    prereg_path = args.out_dir / "preregistration.json"
    shutil.copyfile(args.preregistration, prereg_path)
    prereg = load(prereg_path)
    pair_root = args.out_dir / args.pair_id
    pair_root.mkdir()
    parent_raw = args.out_dir / "raw" / args.pair_id / "parent"
    candidate_raw = args.out_dir / "raw" / args.pair_id / "candidate"
    hardlink_tree(args.parent, parent_raw)
    hardlink_tree(args.candidate, candidate_raw)
    parent = summarize_arm(parent_raw, "parent", prereg)
    candidate = summarize_arm(candidate_raw, "candidate", prereg)
    ns04 = "NOT_RUN"
    if args.ns04_parent is not None or args.ns04_candidate is not None:
        require(args.ns04_parent is not None and args.ns04_candidate is not None,
                "NS04 requires both parent and candidate")
        ns04_parent_raw = args.out_dir / "raw" / "ns04" / "parent"
        ns04_candidate_raw = args.out_dir / "raw" / "ns04" / "candidate"
        hardlink_tree(args.ns04_parent, ns04_parent_raw)
        hardlink_tree(args.ns04_candidate, ns04_candidate_raw)
        ns04_parent = summarize_ns04(ns04_parent_raw)
        ns04_candidate = summarize_ns04(ns04_candidate_raw)
        retained = []
        for index, path in enumerate(args.ns04_retained):
            destination = args.out_dir / "raw" / "ns04" / f"retained-{index}"
            hardlink_tree(path, destination)
            retained.append(summarize_ns04(destination))
        ns04 = {
            "arms": {"parent": ns04_parent, "candidate": ns04_candidate},
            "retained_attempts": retained,
            "paired": {
                "played_pcm_exact": ns04_parent["played_pcm_sha256"] ==
                    ns04_candidate["played_pcm_sha256"],
                "acknowledged_cursor_exact": ns04_parent["acknowledged_cursor"] ==
                    ns04_candidate["acknowledged_cursor"],
                "engineering_pass": ns04_parent["engineering_pass"] and
                    ns04_candidate["engineering_pass"],
            },
            "candidate_hard_thresholds": ns04_candidate["hard_thresholds"],
            "overall": "PASS" if all(value == "PASS" for value in
                ns04_candidate["hard_thresholds"].values()) else "NOT_MET",
            "physical_microphone_speaker": "NOT_RUN",
            "acoustic_perceptual_evaluation": "NOT_RUN",
        }
    public_pcm = "NOT_RUN"
    if args.public_pcm is not None:
        public_raw = args.out_dir / "raw" / "public-pcm"
        hardlink_tree(args.public_pcm, public_raw)
        public_pcm = summarize_public_pcm(public_raw)
    network = "NOT_RUN"
    if args.network is not None:
        network_raw = args.out_dir / "raw" / "network"
        hardlink_tree(args.network, network_raw)
        network = summarize_network(network_raw)
    for log in args.log:
        name, separator, path = log.partition("=")
        require(separator and re.fullmatch(r"[a-z0-9_.-]+", name) and Path(path).is_file(),
                "invalid --log NAME=PATH")
        destination = args.out_dir / "logs" / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, destination)
    negative_attempts = []
    for negative in args.negative:
        name, separator, path = negative.partition("=")
        require(separator and re.fullmatch(r"[a-z0-9_.-]+", name) and Path(path).is_dir(),
                "invalid --negative NAME=PATH")
        destination = args.out_dir / "raw" / "negative" / name
        hardlink_tree(path, destination)
        negative_attempts.append({"name": name, **tree_identity(destination)})
    exact_text = parent["clean_text_sha256"] == candidate["clean_text_sha256"]
    scorecard = {
        "format": FORMAT, "pair_id": args.pair_id,
        "official_base_commit": prereg["official_base_commit"],
        "preregistration_sha256": file_sha(prereg_path)["sha256"],
        "physical_gpu": prereg["physical_gpu"],
        "arms": {"parent": parent, "candidate": candidate},
        "paired": {"clean_text_exact": exact_text,
                   "request_exact": parent["request_sha256"] == candidate["request_sha256"],
                   "candidate_correctness_gain":
                       candidate["all_candidate_correctness_gates"] and
                       not parent["all_candidate_correctness_gates"]},
        "ns01_hard_thresholds": candidate["hard_latency"],
        "ns01_visual_and_total_completion": "PASS" if
            candidate["ideal_single_stream_correctness"] else "NOT_MET",
        "ns04_engineering": ns04,
        "public_pcm_regression": public_pcm,
        "network_regression": network,
        "negative_attempts": negative_attempts,
        "physical_microphone_speaker": "NOT_RUN",
        "acoustic_perceptual_evaluation": "NOT_RUN",
        "training_optimizer": "NOT_RUN",
        "resident_service_deployment": "NOT_RUN",
        "promotion_scope": "TTS/stream correctness only; never implies full NS-01/NS-04 PASS",
        "promotion_decision": "PROMOTED_CORRECTNESS" if
            exact_text and candidate["all_candidate_correctness_gates"] else "NOT_PROMOTED",
        "evidence_policy": "external raw media/frames; complete self-excluded manifest; no tracked per-event blobs",
    }
    score_path = args.out_dir / "scorecard.json"
    score_path.write_bytes(canonical(scorecard) + b"\n")
    files = sorted(path for path in args.out_dir.rglob("*")
                   if path.is_file() and path.name != "SHA256SUMS")
    (args.out_dir / "SHA256SUMS").write_text("".join(
        f"{file_sha(path)['sha256']}  {path.relative_to(args.out_dir)}\n" for path in files))
    print(json.dumps({"promotion_decision": scorecard["promotion_decision"],
                      "scorecard_sha256": file_sha(score_path)["sha256"],
                      "manifest_sha256": file_sha(args.out_dir / "SHA256SUMS")["sha256"],
                      "manifest_files": len(files)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
