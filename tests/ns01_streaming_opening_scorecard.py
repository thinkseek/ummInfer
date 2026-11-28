#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the compact NS-01 streaming-opening paired scorecard from raw arms."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import wave


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


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                    ensure_ascii=False, allow_nan=False) + "\n")


def load(path):
    return json.loads(Path(path).read_text())


def milliseconds(value):
    return None if value is None else round(value, 3)


def native_log_rows(path):
    rows = {}
    if not Path(path).exists():
        return rows
    pattern = re.compile(
        r"tts request request_id=(\S+).*?codec_frames=(\d+) natural_eos=(\d+)"
        r" samples=(\d+).*?elapsed_ms=(\d+)")
    for line in Path(path).read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            rows[match.group(1)] = {
                "codec_frames": int(match.group(2)),
                "natural_eos": match.group(3) == "1",
                "samples": int(match.group(4)),
                "native_elapsed_ms": int(match.group(5)),
            }
    return rows


def vlm_finish(path):
    if not Path(path).exists():
        return {"finish_reason": "MISSING", "generated_tokens": None}
    matches = list(re.finditer(
        r"qwen38 request request_id=\S+.*?generated_tokens=(\d+) finish_reason=(\S+)",
        Path(path).read_text(errors="replace")))
    if not matches:
        return {"finish_reason": "MISSING", "generated_tokens": None}
    match = matches[-1]
    return {"generated_tokens": int(match.group(1)), "finish_reason": match.group(2)}


def content_projection(arm, prereg):
    text = arm["clean_text"]
    case = prereg["golden_cases"][arm["case"]]
    headings = list(re.finditer(r"(?m)^ {0,3}#{1,6}[ \t]+", text))
    first_heading = headings[0].start() if headings else len(text)
    before_heading = text[:first_heading]
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", before_heading)
                  if part.strip()]
    slots = arm["visual_slots"]
    image_slot = next((slot for slot in slots if slot.get("slot_id") == "slot-0"), None)
    video_slot = next((slot for slot in slots if slot.get("slot_id") == "slot-1"), None)
    opening_bytes = int(image_slot.get("text_offset", 0)) if image_slot else 0
    opening = text.encode()[:opening_bytes].decode(errors="ignore").strip()
    chinese_count = len(re.findall(r"[㐀-鿿]", before_heading))
    english_words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", before_heading)
    opening_english_words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", opening)
    opening_chinese = len(re.findall(r"[㐀-鿿]", opening))
    opening_units = opening_chinese if arm["case"] == "chinese" else len(opening_english_words)
    term_hits = [term for term in case["required_terms"]
                 if term.lower() in text.lower()]
    return {
        "clean_text_bytes": len(text.encode()), "clean_text_sha256": sha(text.encode()),
        "heading_count": len(headings), "opening_paragraphs_before_heading": len(paragraphs),
        "opening_chinese_codepoints_before_heading": chinese_count,
        "opening_english_words_before_heading": len(english_words),
        "opening": opening, "opening_units": opening_units,
        "opening_before_image": bool(opening) and opening_bytes > 0,
        "opening_is_concise": 0 < opening_units <= prereg["content_and_lifecycle_gates"]
                                                   ["opening_max_words_or_chinese_codepoints"],
        "opening_has_no_heading": not re.search(r"(?m)^\s*#", opening),
        "term_hits": term_hits,
        "tag_leaks": len(re.findall(r"\[\[\s*(?:image|video|visual)\s*:|source=slot-\d+",
                                    text, flags=re.I)),
        "image_slot_present": image_slot is not None,
        "video_slot_present": video_slot is not None,
        "video_explicit_anchor": bool(video_slot and video_slot.get("anchor_explicit") is True
                                      and video_slot.get("anchor_slot_index") == 0),
        "minimum_clean_text_bytes_met": len(text.encode()) >= case["minimum_clean_text_bytes"],
        "minimum_opening_paragraphs_met": len(paragraphs) >=
                                          case["minimum_opening_paragraphs_before_first_heading"],
        "minimum_background_met": (chinese_count >= case.get(
            "minimum_opening_chinese_codepoints_before_first_heading", 0) and
            len(english_words) >= case.get("minimum_opening_english_words_before_first_heading", 0)),
        "minimum_headings_met": len(headings) >= case["minimum_markdown_heading_lines"],
        "minimum_terms_met": len(term_hits) >= case["minimum_required_terms"],
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


def pcm_and_legs(arm_dir, arm):
    taps = load(arm_dir / "native-taps.json")
    outputs = taps["tts_output"]
    log_rows = native_log_rows(arm_dir / "tts.log")
    public_audio = []
    record = load(arm_dir / "public-record.json")
    for artifact in record["artifacts"]:
        if artifact["media_type"] == "audio":
            raw = (arm_dir / "capture" / "artifacts" / artifact["ref"]).read_bytes()
            public_audio.append((artifact, raw))
    aggregate = b"".join(raw for _, raw in public_audio)
    joins = {item["request_id"]: item for item in arm.get("tts_legs", [])}
    legs = []
    public_cursor = 0
    for ordinal, submit in enumerate(taps["tts"]):
        request_id = submit["request_id"]
        native_streams = [item for item in outputs
                          if item.get("message_type") == "stream"
                          and item.get("request_id") == request_id]
        terminals = [item for item in outputs
                     if item.get("message_type") == "complete"
                     and item.get("request_id") == request_id]
        native_pcm = bytearray()
        native_exact = bool(native_streams)
        for expected, item in enumerate(native_streams):
            packed = __import__("msgpack").unpackb(
                (arm_dir / item["packed_file"]).read_bytes(), raw=False)
            raw = packed["chunk"]["audio"]
            native_exact = native_exact and item.get("stream_index") == expected
            native_exact = native_exact and sha(raw) == item["audio_sha256"]
            native_pcm.extend(raw)
        public_exact = True
        # Recompute the public join without retaining per-event rows in the scorecard.
        native_offset = 0
        for expected, native in enumerate(native_streams):
            packed = __import__("msgpack").unpackb(
                (arm_dir / native["packed_file"]).read_bytes(), raw=False)
            native_raw = packed["chunk"]["audio"]
            if public_cursor >= len(public_audio):
                public_exact = False
                continue
            artifact, public_raw = public_audio[public_cursor]
            public_exact = public_exact and artifact.get("stream_index") == expected \
                and public_raw == native_raw
            native_offset += len(native_raw)
            public_cursor += 1
        terminal_wav_exact = False
        terminal_success = False
        if len(terminals) == 1:
            terminal_success = terminals[0].get("success") is True
            packed = __import__("msgpack").unpackb(
                (arm_dir / terminals[0]["packed_file"]).read_bytes(), raw=False)
            result = packed.get("result") or {}
            wav_raw = result.get("data")
            if terminal_success and isinstance(wav_raw, bytes):
                import io
                with wave.open(io.BytesIO(wav_raw), "rb") as wav:
                    terminal_wav_exact = wav.getnchannels() == 1 and wav.getsampwidth() == 2 \
                        and wav.getframerate() == 24000 and wav.readframes(wav.getnframes()) == bytes(native_pcm)
        log = log_rows.get(request_id, {"codec_frames": None, "natural_eos": False,
                                       "samples": None, "native_elapsed_ms": None})
        join = joins.get(request_id, {})
        legs.append({
            "ordinal": ordinal, "request_id": request_id,
            "text_sha256": submit["text_sha256"], "chunk_count": len(native_streams),
            "pcm_bytes": len(native_pcm), "pcm_sha256": sha(bytes(native_pcm)),
            "native_chunk_order_exact": native_exact,
            "public_pcm_exact": public_exact and native_offset == len(native_pcm),
            "native_terminal_count": len(terminals), "native_terminal_success": terminal_success,
            "terminal_wav_exact": terminal_wav_exact,
            "producer_join_exact": join.get("exact_native_public_wav_join", False),
            **log,
        })
    tts_text = "".join(item["text"] for item in taps["tts"])
    return legs, {"public_chunks": len(public_audio), "public_bytes": len(aggregate),
                  "public_sha256": sha(aggregate),
                  "unowned_public_chunks": len(public_audio) - public_cursor,
                  "first_tts_text": taps["tts"][0]["text"] if taps["tts"] else None,
                  "tts_text_matches_clean": spoken_text_projection(tts_text) ==
                                            spoken_text_projection(arm["clean_text"]),
                  "native_tts_tag_leaks": len(re.findall(
                      r"\[\[\s*(?:image|video|visual)\s*:|source=slot-\d+", tts_text, flags=re.I)),
                  "native_tts_markdown_presentation_leaks": sum(
                      native_speech_has_presentation_tokens(item["text"]) for item in taps["tts"])}


def arm_summary(arm_dir, prereg):
    arm_dir = Path(arm_dir)
    arm = load(arm_dir / "arm.json")
    content = content_projection(arm, prereg)
    legs, pcm = pcm_and_legs(arm_dir, arm)
    timings = arm["interactive_timings"]
    origin = timings["capture_origin_monotonic_ns"]
    video_submits = arm["video_submits"]
    i2v_submit_ms = None if not video_submits else (
        video_submits[0]["observed_monotonic_ns"] - origin) / 1e6
    measurements = {
        "ttft_ms": milliseconds(timings["ttft_ns"]),
        "last_vlm_text_ms": milliseconds(timings["last_vlm_delta_ns"]),
        "asr": "NOT_APPLICABLE_TEXT_INPUT",
        "first_tts_submit_ms": milliseconds(arm["first_tts_submit_ms"]),
        "ttfa_ms": milliseconds(timings["ttfa_ns"]),
        "image_slot_ms": milliseconds(timings["image_slot_ns"]),
        "first_image_ms": milliseconds(timings["first_image_ns"]),
        "video_slot_ms": milliseconds(timings["video_slot_ns"]),
        "i2v_native_submit_ms": milliseconds(i2v_submit_ms),
        "i2v_first_public_chunk_ms": milliseconds(timings["video_first_public_chunk_ns"]),
        "i2v_playable_ms": milliseconds(timings["video_public_complete_ns"]),
    }
    lifecycle = arm["stage_lifecycle"]
    for name, tick in (("tts_ready", "tts_ready_monotonic_ns"),
                       ("wan_ready", "wan_ready_monotonic_ns")):
        measurements[name + "_ms"] = milliseconds(
            None if lifecycle.get(tick) is None else (lifecycle[tick] - origin) / 1e6)
    failures = list(arm["failures"])
    def fail(condition, label):
        if not condition:
            failures.append(label)
    fail(arm["terminal_state"] == "completed", "PUBLIC_TERMINAL_NOT_COMPLETE")
    finish = vlm_finish(arm_dir / "vlm.log")
    fail(finish["finish_reason"] == "stop", "VLM_TRUNCATED_OR_FINISH_REASON_MISSING")
    for key in ("opening_before_image", "opening_is_concise", "opening_has_no_heading",
                "image_slot_present", "video_slot_present", "video_explicit_anchor",
                "minimum_clean_text_bytes_met", "minimum_opening_paragraphs_met",
                "minimum_background_met", "minimum_headings_met", "minimum_terms_met"):
        fail(content[key], "CONTENT_" + key.upper())
    fail(content["tag_leaks"] == 0, "PUBLIC_TAG_OR_DIRECTIVE_LEAK")
    fail(arm["source"] is not None, "FIRST_IMAGE_MISSING")
    fail(arm["video"] is not None, "I2V_MEDIA_MISSING")
    fail(arm["video_stream"] is not None, "I2V_NOT_PLAYABLE")
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
    thresholds = prereg["hard_latency_thresholds_ms"]
    hard = {}
    for metric, threshold_key in (("ttft_ms", "ttft_strictly_below"),
                                  ("ttfa_ms", "ttfa_strictly_below"),
                                  ("first_image_ms", "first_image_strictly_below"),
                                  ("i2v_playable_ms", "i2v_playable_strictly_below")):
        value = measurements[metric]
        passed = value is not None and value < thresholds[threshold_key]
        hard[metric] = {"value_ms": value, "strictly_below_ms": thresholds[threshold_key],
                        "result": "MET" if passed else "NOT_MET"}
    provenance = load(arm_dir / "provenance.json")
    model_manifest = load(arm_dir / "model-identity.json")
    model_identity = {}
    for name, model in model_manifest.items():
        config_name = "config.json" if "config.json" in model["files"] else "model_index.json"
        model_identity[name] = {
            "model_id": model["model_id"], "revision": "UNKNOWN",
            "tree_sha256": model["tree_sha256"], "file_count": len(model["files"]),
            "config_sha256": model["files"][config_name]["sha256"]}
    binary_identity = {name: {"sha256": value["sha256"], "bytes": value["bytes"]}
                       for name, value in provenance.items()
                       if name in ("server_binary", "vlm_binary", "tts_binary", "wan_binary")}
    raw = {
        "arm_sha256sums_sha256": file_sha(arm_dir / "SHA256SUMS"),
        "raw_sse_sha256": arm["raw_sse"]["raw_sha256"],
        "raw_sse_bytes": arm["raw_sse"]["raw_bytes"],
        "request_sha256": arm["request_sha256"],
        "server_log_sha256": file_sha(arm_dir / "server.log"),
        "vlm_log_sha256": file_sha(arm_dir / "vlm.log"),
        "tts_log_sha256": file_sha(arm_dir / "tts.log") if (arm_dir / "tts.log").exists() else None,
        "video_log_sha256": file_sha(arm_dir / "video.log") if (arm_dir / "video.log").exists() else None,
    }
    return {
        "case": arm["case"], "repetition": arm["repetition"], "arm": arm["arm"],
        "execution_commit": arm["execution_commit"], "request_sha256": arm["request_sha256"],
        "public_request_id": arm["public_request_id"], "terminal_state": arm["terminal_state"],
        "vlm_finish": finish, "measurements": measurements, "hard_thresholds": hard,
        "content": content, "source": arm["source"], "video": arm["video"],
        "video_stream": arm["video_stream"], "pcm": pcm, "tts_legs": legs,
        "native_video_submits": [{key: value for key, value in item.items()
                                  if key not in ("packed_file",)} for item in video_submits],
        "event_counts": {"public_events": arm["public_event_count"],
                         "public_artifacts": arm["public_artifact_count"],
                         "audio_artifacts": arm["audio_artifact_count"]},
        "failures": sorted(set(failures)), "binary_identity": binary_identity,
        "model_identity": model_identity, "raw": raw,
        "health_final": arm["health_final"], "perceptual_quality": "NOT_RUN",
    }


def median(values):
    values = [value for value in values if value is not None]
    return None if not values else round(statistics.median(values), 3)


def pair_summary(before, after):
    metrics = ("ttft_ms", "first_tts_submit_ms", "ttfa_ms", "image_slot_ms",
               "first_image_ms", "video_slot_ms", "i2v_native_submit_ms", "i2v_playable_ms")
    deltas = {}
    for metric in metrics:
        old = before["measurements"][metric]
        new = after["measurements"][metric]
        delta = None if old is None or new is None else round(new - old, 3)
        deltas[metric] = {"before_ms": old, "after_ms": new, "after_minus_before_ms": delta,
                          "improved": new is not None and (old is None or new < old),
                          "missing_before": old is None, "missing_after": new is None}
    return {"before": before["case"] + "-r" + str(before["repetition"]) + "-before",
            "after": after["case"] + "-r" + str(after["repetition"]) + "-after",
            "timing": deltas,
            "after_failures": after["failures"], "before_failures": before["failures"]}


def infrastructure_attempts(root):
    if root is None:
        return []
    root = Path(root)
    result = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        files = {path.relative_to(directory).as_posix(): file_sha(path)
                 for path in sorted(directory.rglob("*")) if path.is_file()
                 and "run" not in path.relative_to(directory).parts}
        arm_path = directory / "arm.json"
        failures = load(arm_path).get("failures", []) if arm_path.exists() else [
            "pre-request stage startup failure; see retained image/vlm logs"]
        result.append({"name": directory.name, "file_count": len(files),
                       "tree_sha256": sha(canonical(files)), "failures": failures,
                       "accepted_measurement": False,
                       "reason": "infrastructure-only; no accepted public measurement or candidate tuning"})
    return result


def build(args):
    bundle = Path(args.bundle)
    prereg = load(args.preregistration)
    candidate = load(args.candidate)
    arms = {}
    for case in ("chinese", "english"):
        for repetition in (1, 2):
            for arm in ("before", "after"):
                name = f"{case}-r{repetition}-{arm}"
                arms[name] = arm_summary(bundle / name, prereg)
    pairs = []
    for case in ("chinese", "english"):
        for repetition in (1, 2):
            pairs.append(pair_summary(arms[f"{case}-r{repetition}-before"],
                                      arms[f"{case}-r{repetition}-after"]))
    aggregates = {}
    for case in ("chinese", "english"):
        for arm in ("before", "after"):
            selected = [value for key, value in arms.items()
                        if value["case"] == case and value["arm"] == arm]
            aggregates[f"{case}-{arm}"] = {
                metric: median([item["measurements"][metric] for item in selected])
                for metric in ("ttft_ms", "first_tts_submit_ms", "ttfa_ms", "first_image_ms",
                               "video_slot_ms", "i2v_native_submit_ms", "i2v_playable_ms")}
    rule = candidate["promotion_rule"]
    chinese_ttfa_before = aggregates["chinese-before"]["ttfa_ms"]
    chinese_ttfa_after = aggregates["chinese-after"]["ttfa_ms"]
    chinese_image_before = aggregates["chinese-before"]["first_image_ms"]
    chinese_image_after = aggregates["chinese-after"]["first_image_ms"]
    ttfa_gain = None if chinese_ttfa_before in (None, 0) or chinese_ttfa_after is None else \
        (chinese_ttfa_before - chinese_ttfa_after) / chinese_ttfa_before
    image_gain = None if chinese_image_before in (None, 0) or chinese_image_after is None else \
        (chinese_image_before - chinese_image_after) / chinese_image_before
    integrity_pass = all(not value["failures"] for value in arms.values() if value["arm"] == "after")
    repeated_gain = all(pair["timing"]["ttfa_ms"]["improved"]
                        and pair["timing"]["first_image_ms"]["improved"] for pair in pairs)
    latency_gain = ttfa_gain is not None and ttfa_gain >= rule["minimum_chinese_median_ttfa_reduction_fraction"] \
        and image_gain is not None and image_gain >= rule["minimum_chinese_median_first_image_reduction_fraction"]
    controls = load(args.controls) if args.controls else {"all_required_pass": False,
                                                         "missing": "controls not supplied"}
    promotion = "PROMOTED_ENGINEERING_GAIN" if integrity_pass and repeated_gain and latency_gain \
        and controls.get("all_required_pass") is True else "NOT_PROMOTED"
    hard_not_met = sorted({f"{key}:{metric}" for key, value in arms.items()
                           for metric, result in value["hard_thresholds"].items()
                           if result["result"] == "NOT_MET"})
    scorecard = {
        "format": "umminfer-ns01-streaming-opening-scorecard-v1",
        "official_base_commit": prereg["official_base_commit"],
        "candidate": candidate, "preregistration_sha256": file_sha(args.preregistration),
        "arms": arms, "pairs": pairs, "aggregates": aggregates,
        "paired_improvement": {
            "chinese_median_ttfa_reduction_fraction": ttfa_gain,
            "chinese_median_first_image_reduction_fraction": image_gain,
            "every_pair_ttfa_and_first_image_improved": repeated_gain,
            "after_text_media_lifecycle_integrity_pass": integrity_pass,
            "required_controls_pass": controls.get("all_required_pass") is True,
        },
        "controls": controls, "promotion_decision": promotion,
        "retained_infrastructure_attempts": infrastructure_attempts(args.negative_root),
        "full_ns01": "NOT_MET", "human_perceptual_evaluation": "NOT_RUN",
        "hard_threshold_not_met": hard_not_met,
        "all_failure_count": sum(len(value["failures"]) for value in arms.values()),
        "resident_service": "NOT_CONTACTED", "deployment": "NOT_RUN",
    }
    write_json(args.out, scorecard)
    print(json.dumps({"promotion_decision": promotion, "aggregates": aggregates,
                      "paired_improvement": scorecard["paired_improvement"],
                      "all_failure_count": scorecard["all_failure_count"],
                      "hard_not_met_count": len(hard_not_met)}, sort_keys=True))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--controls")
    parser.add_argument("--negative-root")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
