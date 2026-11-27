#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fresh-process independent verifier for the NS-01 TTS lifecycle bundle.

This intentionally imports neither the live driver nor scorecard producer. It
checks complete external manifest membership/digests and independently joins raw
public SSE to native MessagePack PCM, terminal WAV, and native EOS log evidence.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
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


def verify_manifest(root):
    rows = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = Path(relative)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) and relative != "SHA256SUMS" and
                not path.is_absolute() and ".." not in path.parts and
                relative not in rows, "invalid or duplicate manifest row")
        rows[relative] = digest
    actual = {path.relative_to(root).as_posix(): path for path in root.rglob("*")
              if path.is_file() and path.name != "SHA256SUMS"}
    require(set(actual) == set(rows), "manifest is incomplete or has extra files")
    for relative, digest in rows.items():
        require(file_sha(actual[relative])["sha256"] == digest,
                f"manifest digest mismatch: {relative}")
    return len(rows), file_sha(root / "SHA256SUMS")["sha256"]


def speech_projection(text):
    # Independent linguistic normalization for the frozen Markdown forms.
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^ {0,3}(?:#+[ \t]+|>[ \t]?|[*+-][ \t]+)", "", line)
        if re.fullmatch(r"\s*(?:-{3,}|\*{3,}|_{3,})\s*", line):
            continue
        lines.append(line)
    text = "\n".join(lines)
    for token in ("**", "__", "~~", "`"):
        text = text.replace(token, "")
    return "".join(text.split())


def independent_tree(root):
    files = {path.relative_to(root).as_posix(): file_sha(path)
             for path in sorted(root.rglob("*")) if path.is_file()}
    return {"file_count": len(files), "tree_sha256": sha(canonical(files))}


def parse_public(raw):
    events = []
    done = 0
    cursor = 0
    while cursor < len(raw):
        end = raw.find(b"\n\n", cursor)
        require(end >= 0, "truncated SSE")
        frame = raw[cursor:end]
        lines = []
        for line in frame.split(b"\n"):
            if line.startswith(b"data:"):
                value = line[5:]
                lines.append(value[1:] if value.startswith(b" ") else value)
        payload = b"\n".join(lines)
        if payload == b"[DONE]":
            done += 1
        else:
            value = json.loads(payload)
            if value.get("unified") is not None:
                events.append((cursor, value["unified"]))
        cursor = end + 2
    return events, done


def wav_pcm(raw):
    with wave.open(io.BytesIO(raw), "rb") as wav:
        require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) ==
                (1, 2, 24000), "bad native WAV")
        return wav.readframes(wav.getnframes())


def raw_native(root):
    import msgpack
    submits = []
    outputs = {}
    for path in root.glob("tts-*.msgpack"):
        match = re.fullmatch(r"tts-(\d+)-([0-9a-f]{64})\.msgpack", path.name)
        if not match:
            continue
        require(file_sha(path)["sha256"] == match.group(2), "submit filename digest mismatch")
        message = msgpack.unpackb(path.read_bytes(), raw=False)
        if message.get("type") != "submit":
            continue
        text = message["data"]["request"]["inputs"]["text"]
        submits.append((int(match.group(1)), message["request_id"], text,
                        file_sha(path), path.stat().st_mtime_ns))
    for path in root.glob("tts-output-*.msgpack"):
        match = re.fullmatch(r"tts-output-(\d+)-([0-9a-f]{64})\.msgpack", path.name)
        if not match:
            continue
        require(file_sha(path)["sha256"] == match.group(2), "output filename digest mismatch")
        message = msgpack.unpackb(path.read_bytes(), raw=False)
        if message.get("type") == "stage_ready":
            continue
        outputs.setdefault(message["request_id"], []).append(
            (int(match.group(1)), message, file_sha(path), path.stat().st_mtime_ns))
    submits.sort()
    for rows in outputs.values():
        rows.sort()
    return submits, outputs


def log_outcomes(root):
    text = (root / "tts.log").read_text(encoding="utf-8", errors="replace")
    completed = {}
    aborted = set()
    for line in text.splitlines():
        match = re.search(r"tts request request_id=([^ ]+) (.*)$", line)
        if match:
            completed[match.group(1)] = dict(
                re.findall(r"([a-z0-9_]+)=([^ ]+)", match.group(2)))
        match = re.search(r"tts request aborted request_id=([^ ]+)", line)
        if match:
            aborted.add(match.group(1))
    return completed, aborted, text


def queue_summary(submits, outputs):
    transitions = []
    for ordinal, request_id, _text, _identity, mtime in submits:
        transitions.append((mtime, 0, ordinal, 1))
        for out_ordinal, message, _identity, out_mtime in outputs.get(request_id, []):
            if message.get("type") == "complete":
                transitions.append((out_mtime, 1, out_ordinal, -1))
    active = highwater = 0
    valid = True
    for _mtime, _tie, _ordinal, delta in sorted(transitions):
        active += delta
        if active < 0:
            valid = False
            active = 0
        highwater = max(highwater, active)
    return {"active_highwater": highwater, "final_unsettled": active,
            "ordered_event_join_valid": valid,
            "clock": "filesystem_mtime_ns_diagnostic"}


def verify_arm(root, arm, prereg, label):
    expected = prereg["golden_case"]
    request_path = root / "request.json"
    request = load(request_path)
    require(request["messages"][0]["content"] == expected["user_text"] and
            request["max_tokens"] == expected["max_tokens"] and
            request["temperature"] == expected["temperature"] and
            request["seed"] == expected["seed"] and
            request["narrative"]["visual_mode"] == "auto" and
            request["narrative"]["continuity_mode"] == "i2v_chain" and
            "teacache_thresh" not in request["video"], f"{label}: request mismatch")
    require(file_sha(request_path)["sha256"] == arm["request_sha256"],
            f"{label}: request hash mismatch")
    trajectory_path = root / "capture" / "trajectories.jsonl"
    trajectory_rows = trajectory_path.read_text().splitlines()
    require(len(trajectory_rows) == 1, f"{label}: trajectory count")
    trajectory = json.loads(trajectory_rows[0])
    require(file_sha(trajectory_path)["sha256"] == arm["trajectory_sha256"],
            f"{label}: trajectory hash")
    raw_path = root / "capture" / "artifacts" / trajectory["sse_replay"]["raw_ref"]
    raw = raw_path.read_bytes()
    require(sha(raw) == trajectory["sse_replay"]["raw_sha256"] ==
            arm["raw_response_sha256"] and len(raw) == arm["raw_response_bytes"],
            f"{label}: raw response identity")
    ticks = {event["source_event_index"]: event["source_event_tick"]
             for event in trajectory["events"]}
    # parse_public's frame offset equals the raw capture's source byte offset;
    # map those offsets back to source frame indices without producer metadata.
    offset_to_index = {}
    cursor = 0
    while cursor < len(raw):
        end = raw.find(b"\n\n", cursor)
        require(end >= 0, "frame mapping truncated")
        offset_to_index[cursor] = len(offset_to_index)
        cursor = end + 2
    parsed, done = parse_public(raw)
    events = [(offset_to_index[offset], event) for offset, event in parsed]
    require(done == arm["public_done_count"] == 1 and
            len(events) == arm["public_event_count"], f"{label}: event/DONE count")
    clean = "".join(event.get("text", "") for _index, event in events
                    if event["kind"] == "vlm_delta")
    require(clean == trajectory["clean_text"] and
            sha(clean.encode()) == arm["clean_text_sha256"] and
            len(clean.encode()) == arm["clean_text_bytes"], f"{label}: clean text")
    terminals = [event for _index, event in events
                 if event["kind"] in ("complete", "cancelled", "error")]
    require(len(terminals) == 1 and terminals[0]["kind"] == arm["public_terminal_kind"] and
            trajectory["terminal_state"] == arm["trajectory_terminal_state"],
            f"{label}: terminal")
    audio = []
    slots = []
    first = {"ttft_ms": None, "ttfa_ms": None, "first_image_ms": None,
             "i2v_complete_ms": None, "total_to_terminal_ms": None}
    media_seen = set()
    images = videos = 0
    for index, event in events:
        tick = ticks.get(index)
        kind = event["kind"]
        if kind == "vlm_delta" and event.get("text") and first["ttft_ms"] is None:
            first["ttft_ms"] = tick / 1e6
        elif kind == "visual_slot":
            slots.append({"slot_id": event.get("slot_id"), "stage": event.get("stage")})
        elif kind in ("complete", "cancelled", "error"):
            first["total_to_terminal_ms"] = tick / 1e6
        elif kind == "artifact":
            desc = event["artifact"]
            media = desc.get("media_type")
            if media == "audio":
                audio.append(base64.b64decode(desc["data"], validate=True))
                if first["ttfa_ms"] is None:
                    first["ttfa_ms"] = tick / 1e6
            final = desc["index"] + 1 == desc["total"]
            if final and desc["artifact_id"] not in media_seen:
                media_seen.add(desc["artifact_id"])
                if media == "image":
                    images += 1
                    if desc.get("slot_id") == "slot-0" and first["first_image_ms"] is None:
                        first["first_image_ms"] = tick / 1e6
                elif media == "video":
                    videos += 1
                    first["i2v_complete_ms"] = tick / 1e6
    require(first == arm["timings"], f"{label}: arrival timings")
    limits = {"ttft_ms": 800.0, "ttfa_ms": 800.0,
              "first_image_ms": 2000.0, "i2v_complete_ms": 5000.0}
    hard = {key: "NOT_RUN" if first[key] is None else
            ("PASS" if first[key] < limit else "NOT_MET")
            for key, limit in limits.items()}
    require(hard == arm["hard_latency"], f"{label}: hard thresholds")
    require(slots == arm["visual_slots"] and images == arm["public_image_count"] and
            videos == arm["public_video_count"], f"{label}: visual/media accounting")
    before_heading = re.split(r"(?m)^#", clean, maxsplit=1)[0]
    opening = [part for part in re.split(r"\n\s*\n", before_heading) if part.strip()]
    terms = [term for term in ("accretion disk", "gravitational lensing",
                               "event horizon", "Eddington limit") if term in clean]
    require(len(opening) == arm["opening_paragraphs_before_first_heading"] and
            sum("一" <= c <= "鿿" for c in before_heading) == arm["opening_chinese_codepoints"] and
            sum(line.startswith("#") for line in clean.splitlines()) == arm["heading_lines"] and
            terms == arm["mixed_terms_present"], f"{label}: opening/Markdown accounting")
    submits, outputs = raw_native(root)
    logs, aborted, log_text = log_outcomes(root)
    require(file_sha(root / "tts.log") == arm["native_tts_log"] and
            sha(log_text[-8192:].encode()) == arm["native_log_tail_sha256"],
            f"{label}: native log identity")
    require(queue_summary(submits, outputs) == arm["tts_queue"], f"{label}: queue high-water")
    require(len(submits) == arm["tts_leg_count"], f"{label}: TTS leg count")
    public_cursor = 0
    text_parts = []
    native_success = nonterminal = aborted_count = eos = eos_unknown = exact_wav = exact_public = 0
    presentation_leaks = 0
    for submit_index, (ordinal, request_id, text, submit_identity, _mtime) in enumerate(submits):
        leg = arm["legs"][submit_index]
        text_parts.append(text)
        presentation_leaks += bool(
            re.search(r"(?:\*\*|__|~~|`)", text) or
            re.search(r"(?m)^ {0,3}#+[ \t]+", text) or
            re.search(r"(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", text))
        require(leg["child_id"] == request_id and leg["submit_ordinal"] == ordinal and
                leg["text_sha256"] == sha(text.encode()) and
                leg["text_bytes"] == len(text.encode()) and
                leg["submit_file"] == submit_identity, f"{label}: leg text/submit identity")
        chunks = []
        terminal = []
        for _out_ordinal, message, _identity, _out_mtime in outputs.get(request_id, []):
            if message.get("type") == "stream":
                chunks.append((message["chunk"]["index"], message["chunk"]["audio"]))
            elif message.get("type") == "complete":
                terminal.append(message)
        chunks.sort()
        require([index for index, _pcm in chunks] == list(range(len(chunks))),
                f"{label}: native chunk index gap")
        native_pcm = b"".join(pcm for _index, pcm in chunks)
        public_pcm = b""
        public_count = 0
        exact = True
        for _index, pcm in chunks:
            if public_cursor >= len(audio) or audio[public_cursor] != pcm:
                exact = False
                break
            public_cursor += 1
            public_count += 1
            public_pcm += pcm
        terminal_pcm = None
        wav_sha = None
        success = False
        for message in terminal:
            success = success or message.get("success") is True
            data = (message.get("result") or {}).get("data")
            if isinstance(data, bytes) and data:
                wav_sha = sha(data)
                terminal_pcm = wav_pcm(data)
        fields = logs.get(request_id, {})
        natural = None if "natural_eos" not in fields else fields["natural_eos"] == "1"
        frames = None if "codec_frames" not in fields else int(fields["codec_frames"])
        state = "completed" if success else "aborted" if request_id in aborted else "nonterminal"
        native_success += success
        nonterminal += state == "nonterminal"
        aborted_count += state == "aborted"
        eos += natural is True
        eos_unknown += natural is None
        exact_wav += terminal_pcm is not None and terminal_pcm == native_pcm
        exact_public += exact
        expected_leg = {
            "child_id": request_id, "submit_ordinal": ordinal,
            "text_sha256": sha(text.encode()), "text_bytes": len(text.encode()),
            "native_stream_chunks": len(chunks), "native_pcm_bytes": len(native_pcm),
            "native_pcm_sha256": sha(native_pcm), "native_terminal_count": len(terminal),
            "native_terminal_success": success,
            "native_stream_terminal_pcm_exact": None if terminal_pcm is None else
                terminal_pcm == native_pcm,
            "terminal_wav_sha256": wav_sha, "public_chunks": public_count,
            "public_pcm_bytes": len(public_pcm), "public_pcm_sha256": sha(public_pcm),
            "native_public_pcm_exact": exact, "natural_eos": natural,
            "generated_frames": frames, "state": state,
            "native_output_file_count": len(outputs.get(request_id, [])),
            "submit_file": submit_identity,
        }
        require(leg == expected_leg, f"{label}: per-leg PCM/EOS mismatch at {request_id}")
    all_public = b"".join(audio)
    tag_leaks = sum(clean.count(tag) + "".join(text_parts).count(tag) for tag in TAGS)
    projection_exact = speech_projection("".join(text_parts)) == speech_projection(clean)
    expected_counts = {
        "native_terminal_success_count": native_success,
        "native_nonterminal_count": nonterminal, "native_aborted_count": aborted_count,
        "natural_eos_count": eos, "natural_eos_unknown_count": eos_unknown,
        "native_stream_terminal_exact_count": exact_wav,
        "native_public_exact_leg_count": exact_public,
        "public_pcm_chunks": len(audio), "public_pcm_bytes": len(all_public),
        "public_pcm_sha256": sha(all_public),
        "unmatched_public_pcm_chunks": len(audio) - public_cursor,
        "tag_leak_count": tag_leaks,
        "native_markdown_presentation_leak_count": presentation_leaks,
        "speech_text_projection_exact": projection_exact,
    }
    for key, value in expected_counts.items():
        require(arm[key] == value, f"{label}: aggregate mismatch {key}")
    all_correct = bool(submits and native_success == len(submits) and nonterminal == 0 and
                       eos == len(submits) and exact_wav == len(submits) and
                       exact_public == len(submits) and public_cursor == len(audio) and
                       terminals[0]["kind"] == "complete" and
                       trajectory["terminal_state"] == "completed" and
                       tag_leaks == 0 and presentation_leaks == 0 and projection_exact)
    require(arm["all_candidate_correctness_gates"] == all_correct,
            f"{label}: candidate correctness recomputation")
    ideal = bool(all_correct and
                 [(item["slot_id"], item["stage"]) for item in slots] ==
                 [("slot-0", "image"), ("slot-1", "video")] and images == videos == 1 and
                 len(clean.encode()) >= expected["minimum_clean_text_bytes"] and
                 len(opening) >= expected["minimum_opening_paragraphs_before_first_heading"] and
                 sum("一" <= c <= "鿿" for c in before_heading) >=
                     expected["minimum_opening_chinese_codepoints_before_first_heading"] and
                 sum(line.startswith("#") for line in clean.splitlines()) >=
                     expected["required_markdown_heading_lines"] and
                 len(terms) >= expected["required_mixed_terms_any"])
    require(arm["ideal_single_stream_correctness"] == ideal, f"{label}: ideal recomputation")
    if (root / "identity.json").is_file():
        require(arm["identity"] == load(root / "identity.json") and
                arm["identity_sha256"] == file_sha(root / "identity.json")["sha256"],
                f"{label}: source/binary/model identity")
    else:
        require("identity" not in arm and "identity_sha256" not in arm,
                f"{label}: unexpected identity")


def independent_ns04(root):
    root = Path(root)
    evidence_path = root / "full-duplex-barge-in-evidence.json"
    evidence = load(evidence_path)
    raw_path = root / "raw-events.jsonl"
    raw_identity = file_sha(raw_path)
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    rx = []
    markers = []
    previous = -1
    for row in rows:
        raw = row["raw_message"].encode()
        require(sha(raw) == row["raw_sha256"] and row["observed_monotonic_ns"] >= previous,
                "NS04 raw hash/order")
        previous = row["observed_monotonic_ns"]
        event = json.loads(raw)
        if row["direction"] == "rx":
            rx.append((row["observed_monotonic_ns"], event))
        elif row["direction"] == "marker":
            markers.append((row["observed_monotonic_ns"], event))
    require(raw_identity == evidence["raw_events"] and
            len(rows) == evidence["raw_record_count"] and
            len(rx) == evidence["raw_event_count"], "NS04 evidence/raw count")
    first_id, second_id = evidence["first_response_id"], evidence["second_response_id"]
    first_start = next(at for at, event in markers
                       if event == {"type": "client_barge_start", "response_id": first_id})
    second_start = next(at for at, event in markers
                        if event == {"type": "client_barge_start", "response_id": second_id})
    first_cancel = next((at, event) for at, event in rx
                        if event.get("type") == "response.cancelled" and
                        event.get("response_id") == first_id)
    second_cancel = next((at, event) for at, event in rx
                         if event.get("type") == "response.cancelled" and
                         event.get("response_id") == second_id)
    silence = next(at for at, event in markers
                   if event == {"type": "digital_silence", "response_id": first_id})
    transcripts = [at for at, event in rx if event.get("type") == "transcription.completed"]
    second_audio = next(at for at, event in rx
                        if event.get("type") == "audio.delta" and
                        event.get("response_id") == second_id)
    first_audio = [event for at, event in rx if at <= first_cancel[0] and
                   event.get("type") == "audio.delta" and event.get("response_id") == first_id]
    played_count = evidence["played_audio_events"]
    played = b"".join(base64.b64decode(event["audio"], validate=True)
                      for event in first_audio[:played_count])
    pending = first_audio[played_count:]
    pending_bytes = sum(len(base64.b64decode(event["audio"], validate=True)) for event in pending)
    require(played_count <= len(first_audio) and sha(played) == evidence["played_pcm_sha256"] and
            len(played) == evidence["played_pcm_bytes"] and
            len(pending) == evidence["cancel_pending_audio_events"] and
            pending_bytes == evidence["cancel_pending_pcm_bytes"] and
            first_cancel[1].get("text_cursor") == evidence["acknowledged_cursor"] and
            len(transcripts) >= 2, "NS04 PCM/ACK reconstruction")
    stale_types = {"audio.delta", "audio.done", "response.text.delta", "response.playback_ack",
                   "response.completed", "response.failed"}
    stale = sum(at > cancel[0] and event.get("response_id") == response_id and
                event.get("type") in stale_types for at, event in rx
                for cancel, response_id in ((first_cancel, first_id), (second_cancel, second_id)))
    timings = {"first_interrupt_ms": (first_cancel[0] - first_start) / 1e6,
               "cancel_to_digital_silence_ms": (silence - first_cancel[0]) / 1e6,
               "post_barge_ttfa_from_asr_final_ms": (second_audio - transcripts[1]) / 1e6,
               "barge_to_new_audio_ms": (second_audio - first_start) / 1e6,
               "second_interrupt_ms": (second_cancel[0] - second_start) / 1e6}
    limits = {"first_interrupt_ms": 300.0, "cancel_to_digital_silence_ms": 300.0,
              "post_barge_ttfa_from_asr_final_ms": 800.0,
              "barge_to_new_audio_ms": 800.0, "second_interrupt_ms": 300.0}
    thresholds = {key: "PASS" if value < limits[key] else "NOT_MET"
                  for key, value in timings.items()}
    engineering = bool(evidence.get("status") == "PASS" and not evidence.get("error") and
                       stale == 0 and evidence["final_health"].get("in_flight") == 0 and
                       pending_bytes > 0 and evidence["gpu_postflight"][
                           str(evidence["gpus"]["asr"])]["memory_used_mib"] <= 64 and
                       evidence["gpu_postflight"][str(evidence["gpus"]["asr"])]["utilization_percent"] == 0 and
                       not evidence["gpu_postflight"][str(evidence["gpus"]["asr"])]["compute_processes"])
    return {"status": evidence["status"], "evidence_sha256": file_sha(evidence_path)["sha256"],
            "raw_events": raw_identity, "raw_records": len(rows),
            "first_transcript": evidence["first_transcript"],
            "post_barge_transcript": evidence["post_barge_in_transcript"],
            "acknowledged_cursor": evidence["acknowledged_cursor"],
            "played_pcm_bytes": len(played), "played_pcm_sha256": sha(played),
            "pending_pcm_bytes": pending_bytes, "pending_audio_events": len(pending),
            "stale_output_events": stale, "final_in_flight": evidence["final_health"]["in_flight"],
            "timings_ms": timings, "hard_thresholds": thresholds,
            "engineering_pass": engineering, "physical_microphone_speaker": "NOT_RUN",
            "perceptual_quality": "NOT_RUN"}


def independent_public_pcm(root):
    root = Path(root)
    verify_manifest(root)
    first = root / "verification-r1.json"
    second = root / "verification-r2.json"
    require(first.read_bytes() == second.read_bytes(), "public fresh reports differ")
    report = load(first)
    capture = load(root / "capture-manifest.json")
    delayed = report["realtime"]["delayed_read"]
    counts = report["counts"]
    final_acks = sum(row["final_playback_ack"].get("final") is True
                     for row in report["realtime"]["responses"])
    require(report["status"] == "PASS" and capture["physical_gpu"] == 7 and
            counts["realtime_responses"] == 29 and final_acks == 29 and
            delayed["paused_readers"] == 29 and
            delayed["native_audio_events_during_pause"] == 4096 and
            delayed["public_audio_events_during_pause"] == 0 and
            delayed["all_pcm_and_cursors_drained_exact"] and
            report["cancel_retry"]["retry_matches_fresh_repeat"] and
            report["cleanup"]["final_in_flight"] == 0 and
            report["cleanup"]["owned_processes_exited"], "public PCM raw gate incomplete")
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


def independent_network(root):
    root = Path(root)
    verify_manifest(root)
    report_path = root / "independent-verification.json"
    run_path = root / "run.json"
    report = load(report_path)
    run = load(run_path)
    require(report["status"] == "PASS" and report["sessions"] == 6 and
            report["rounds"] == 3 and report["connections"] == 15 and
            report["responses"] == 41 and report["verified_next_turn_prompts"] == 30 and
            report["terminal_callbacks"] == 32 and report["stale_events"] == 0 and
            report["cross_session_leaks"] == 0 and report["final_in_flight"] == 0 and
            run["cleanup"]["selected_gpus_idle"], "network raw gate incomplete")
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
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    require(not args.out.exists() and root not in args.out.resolve().parents,
            "verification output must be new and outside the sealed bundle")
    file_count, manifest_sha = verify_manifest(root)
    score_path = root / "scorecard.json"
    score = load(score_path)
    require(score["format"] == FORMAT, "unknown scorecard format")
    prereg = load(root / "preregistration.json")
    require(score["preregistration_sha256"] ==
            file_sha(root / "preregistration.json")["sha256"] and
            score["official_base_commit"] == prereg["official_base_commit"] and
            score["physical_gpu"] == prereg["physical_gpu"], "scorecard identity")
    for label in ("parent", "candidate"):
        verify_arm(root / "raw" / score["pair_id"] / label,
                   score["arms"][label], prereg, label)
    ns04 = score["ns04_engineering"]
    if ns04 != "NOT_RUN":
        ns04_parent = independent_ns04(root / "raw" / "ns04" / "parent")
        ns04_candidate = independent_ns04(root / "raw" / "ns04" / "candidate")
        require(ns04["arms"] == {"parent": ns04_parent, "candidate": ns04_candidate},
                "NS04 raw summary mismatch")
        retained = [independent_ns04(root / "raw" / "ns04" / f"retained-{index}")
                    for index in range(len(ns04["retained_attempts"]))]
        require(retained == ns04["retained_attempts"], "NS04 retained attempt mismatch")
        require(ns04["paired"] == {
            "played_pcm_exact": ns04_parent["played_pcm_sha256"] ==
                ns04_candidate["played_pcm_sha256"],
            "acknowledged_cursor_exact": ns04_parent["acknowledged_cursor"] ==
                ns04_candidate["acknowledged_cursor"],
            "engineering_pass": ns04_parent["engineering_pass"] and
                ns04_candidate["engineering_pass"]}, "NS04 paired mismatch")
        require(ns04["candidate_hard_thresholds"] == ns04_candidate["hard_thresholds"] and
                ns04["overall"] == ("PASS" if all(value == "PASS" for value in
                    ns04_candidate["hard_thresholds"].values()) else "NOT_MET") and
                ns04["physical_microphone_speaker"] == "NOT_RUN" and
                ns04["acoustic_perceptual_evaluation"] == "NOT_RUN", "NS04 threshold boundary")
    public_pcm = score["public_pcm_regression"]
    if public_pcm != "NOT_RUN":
        require(independent_public_pcm(root / "raw" / "public-pcm") == public_pcm,
                "public PCM independent summary mismatch")
    network = score["network_regression"]
    if network != "NOT_RUN":
        require(independent_network(root / "raw" / "network") == network,
                "network independent summary mismatch")
    for negative in score["negative_attempts"]:
        expected = {"name": negative["name"],
                    **independent_tree(root / "raw" / "negative" / negative["name"])}
        require(expected == negative, "negative-attempt tree mismatch")
    parent = score["arms"]["parent"]
    candidate = score["arms"]["candidate"]
    text_exact = parent["clean_text_sha256"] == candidate["clean_text_sha256"]
    require(score["paired"] == {
        "clean_text_exact": text_exact,
        "request_exact": parent["request_sha256"] == candidate["request_sha256"],
        "candidate_correctness_gain": candidate["all_candidate_correctness_gates"] and
            not parent["all_candidate_correctness_gates"]}, "paired decision mismatch")
    require(score["ns01_hard_thresholds"] == candidate["hard_latency"] and
            score["ns01_visual_and_total_completion"] ==
                ("PASS" if candidate["ideal_single_stream_correctness"] else "NOT_MET"),
            "NS-01 boundary mismatch")
    promoted = text_exact and candidate["all_candidate_correctness_gates"]
    require(score["promotion_decision"] ==
            ("PROMOTED_CORRECTNESS" if promoted else "NOT_PROMOTED"),
            "promotion decision mismatch")
    require(score["physical_microphone_speaker"] == "NOT_RUN" and
            score["acoustic_perceptual_evaluation"] == "NOT_RUN" and
            score["training_optimizer"] == "NOT_RUN" and
            score["resident_service_deployment"] == "NOT_RUN", "NOT_RUN boundary changed")
    result = {"status": "PASS", "scorecard_sha256": file_sha(score_path)["sha256"],
              "manifest_sha256": manifest_sha, "manifest_files": file_count,
              "promotion_decision": score["promotion_decision"],
              "native_public_pcm_join": "PASS", "natural_eos_recomputed": "PASS",
              "thresholds_recomputed": "PASS",
              "ns04_raw_recomputed": "NOT_RUN" if score["ns04_engineering"] == "NOT_RUN" else "PASS",
              "public_pcm_recomputed": "NOT_RUN" if score["public_pcm_regression"] == "NOT_RUN" else "PASS",
              "network_recomputed": "NOT_RUN" if score["network_regression"] == "NOT_RUN" else "PASS",
              "physical_perceptual": "NOT_RUN"}
    args.out.write_bytes(canonical(result) + b"\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
