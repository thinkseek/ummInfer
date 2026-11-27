# SPDX-License-Identifier: Apache-2.0
"""Deterministic canned narrative responses for the RL contract tests."""

import base64
import copy
import hashlib
import json

PNG = b"\x89PNG\r\n\x1a\n" + b"rl-narrative-image"
MP4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"rl-video"
PCM = b"\x01\x00\x02\x00" * 32
WAV = b"RIFF\x20\x00\x00\x00WAVEfmt " + b"rl-input-audio"
FIRST_TEXT = "At dawn, "
SECOND_TEXT = "a red lantern rises."
CLEAN_TEXT = FIRST_TEXT + SECOND_TEXT


def envelope(seq, kind, text="", **fields):
    unified = {
        "seq": seq, "kind": kind, "request_id": "http-narrative-1",
        "session_id": "narrative-rl-fixture", "stage": "vlm",
        "is_final": kind in ("complete", "error", "cancelled"),
        "text": text, **fields,
    }
    return unified


def visual_slot(seq, index, media_type, prompt, text_offset, *,
                continuity_mode=None, anchor_slot_index=None,
                anchor_explicit=None):
    fields = {}
    if continuity_mode is not None:
        fields["continuity_mode"] = continuity_mode
    if anchor_slot_index is not None:
        fields["anchor_slot_index"] = anchor_slot_index
    if anchor_explicit is not None:
        fields["anchor_explicit"] = anchor_explicit
    return envelope(
        seq, "visual_slot", stage=media_type, slot_id=f"slot-{index}",
        slot_index=index, media_type=media_type, prompt=prompt,
        text_offset=text_offset, **fields)


def artifact(seq, artifact_id, media_type, mime_type, raw, stage, *,
             index=0, total=1, slot_id=None, slot_index=None, full_raw=None,
             anchor_slot_index=None, anchor_explicit=None, anchor_sha256=None):
    descriptor = {
        "artifact_id": artifact_id, "media_type": media_type,
        "mime_type": mime_type, "encoding": "base64",
        "data": base64.b64encode(raw).decode("ascii"),
        "index": index, "total": total,
    }
    if index + 1 == total:
        complete_raw = raw if full_raw is None else full_raw
        descriptor["bytes"] = len(complete_raw)
        descriptor["sha256"] = hashlib.sha256(complete_raw).hexdigest()
    if anchor_slot_index is not None:
        descriptor["anchor_slot_index"] = anchor_slot_index
    if anchor_explicit is not None:
        descriptor["anchor_explicit"] = anchor_explicit
    if anchor_sha256 is not None:
        descriptor["anchor_sha256"] = anchor_sha256
    if media_type == "audio":
        descriptor.update({"format": "pcm16", "sample_rate": 24000,
                           "channels": 1, "stream_index": 0})
    else:
        descriptor.update({
            "format": "png" if media_type == "image" else "mp4",
            "size": "512x512" if media_type == "image" else "480x320",
            "seed": 42, "slot_id": slot_id, "slot_index": slot_index,
        })
    return envelope(seq, "artifact", stage=stage, is_final=True,
                    artifact=descriptor)


def narrative_events():
    image_split = len(PNG) // 2
    audio_split = len(PCM) // 2
    return [
        envelope(0, "vlm_delta", FIRST_TEXT),
        visual_slot(1, 0, "image", "a glowing red lantern", len(FIRST_TEXT)),
        artifact(2, "art-image-0", "image", "image/png", PNG[:image_split],
                 "image", index=0, total=2, slot_id="slot-0", slot_index=0),
        visual_slot(3, 1, "video", "the lantern drifts downstream",
                    len(FIRST_TEXT)),
        artifact(4, "art-audio-0", "audio", "audio/pcm", PCM[:audio_split],
                 "tts", index=0, total=2),
        envelope(5, "vlm_delta", SECOND_TEXT),
        artifact(6, "art-video-1", "video", "video/mp4", MP4, "video",
                 slot_id="slot-1", slot_index=1),
        artifact(7, "art-image-0", "image", "image/png", PNG[image_split:],
                 "image", index=1, total=2, slot_id="slot-0", slot_index=0,
                 full_raw=PNG),
        visual_slot(8, 2, "image", "an optional alternate view", len(CLEAN_TEXT)),
        envelope(9, "leg_error", stage="image", error="render denied",
                 slot_id="slot-2", slot_index=2),
        artifact(10, "art-audio-0", "audio", "audio/pcm", PCM[audio_split:],
                 "tts", index=1, total=2, full_raw=PCM),
        envelope(11, "complete", CLEAN_TEXT, legs=[
            {"stage": "image", "slot_id": "slot-0", "slot_index": 0,
             "state": "completed"},
            {"stage": "video", "slot_id": "slot-1", "slot_index": 1,
             "state": "completed"},
            {"stage": "image", "slot_id": "slot-2", "slot_index": 2,
             "state": "failed", "error": "render denied"},
            {"stage": "tts", "state": "completed"},
        ]),
    ]


def linked_events(*, delayed=False):
    anchor_sha = hashlib.sha256(PNG).hexdigest()
    image_slot = visual_slot(1, 0, "image", "a glowing red lantern",
                             len(FIRST_TEXT), continuity_mode="i2v_chain")
    video_slot = visual_slot(2 if delayed else 3, 1, "video",
                             "the lantern sways in the wind", len(FIRST_TEXT),
                             continuity_mode="i2v_chain", anchor_slot_index=0,
                             anchor_explicit=True)
    image_artifact = artifact(
        3 if delayed else 2, "art-image-linked", "image", "image/png", PNG,
        "image", slot_id="slot-0", slot_index=0)
    events = [
        envelope(0, "vlm_delta", FIRST_TEXT),
        image_slot,
        video_slot if delayed else image_artifact,
        image_artifact if delayed else video_slot,
        artifact(4, "art-video-linked", "video", "video/mp4", MP4, "video",
                 slot_id="slot-1", slot_index=1, anchor_slot_index=0,
                 anchor_explicit=True, anchor_sha256=anchor_sha),
        envelope(5, "vlm_delta", SECOND_TEXT),
        envelope(6, "complete", CLEAN_TEXT, legs=[
            {"stage": "image", "slot_id": "slot-0", "slot_index": 0,
             "state": "completed"},
            {"stage": "video", "slot_id": "slot-1", "slot_index": 1,
             "state": "completed", "anchor_slot_index": 0,
             "anchor_explicit": True, "anchor_sha256": anchor_sha},
        ]),
    ]
    return events


def cross_slot_events():
    """Image establishment -> exact image edit -> exact edited-image I2V."""
    edited_png = PNG + b"-edited"
    source0_sha = hashlib.sha256(PNG).hexdigest()
    source1_sha = hashlib.sha256(edited_png).hexdigest()
    return [
        envelope(0, "vlm_delta", FIRST_TEXT),
        visual_slot(1, 0, "image", "establish the character", len(FIRST_TEXT),
                    continuity_mode="i2v_chain"),
        artifact(2, "art-cross-0", "image", "image/png", PNG, "image",
                 slot_id="slot-0", slot_index=0),
        visual_slot(3, 1, "image", "same character at night", len(FIRST_TEXT),
                    continuity_mode="i2v_chain", anchor_slot_index=0,
                    anchor_explicit=True),
        artifact(4, "art-cross-1", "image", "image/png", edited_png, "image",
                 slot_id="slot-1", slot_index=1, anchor_slot_index=0,
                 anchor_explicit=True, anchor_sha256=source0_sha),
        visual_slot(5, 2, "video", "the character turns", len(FIRST_TEXT),
                    continuity_mode="i2v_chain", anchor_slot_index=1,
                    anchor_explicit=True),
        artifact(6, "art-cross-2", "video", "video/mp4", MP4, "video",
                 slot_id="slot-2", slot_index=2, anchor_slot_index=1,
                 anchor_explicit=True, anchor_sha256=source1_sha),
        envelope(7, "vlm_delta", SECOND_TEXT),
        envelope(8, "complete", CLEAN_TEXT, legs=[
            {"stage": "image", "slot_id": "slot-0", "slot_index": 0,
             "state": "completed"},
            {"stage": "image", "slot_id": "slot-1", "slot_index": 1,
             "state": "completed", "anchor_slot_index": 0,
             "anchor_explicit": True, "anchor_sha256": source0_sha},
            {"stage": "video", "slot_id": "slot-2", "slot_index": 2,
             "state": "completed", "anchor_slot_index": 1,
             "anchor_explicit": True, "anchor_sha256": source1_sha},
        ]),
    ]


def cancelled_events():
    source = narrative_events()
    # Keep only the first image/audio chunks; video has completed. Renumber the
    # public server sequence after removing the image's final chunk.
    events = copy.deepcopy([source[index] for index in (0, 1, 2, 3, 4, 5, 6, 8, 9)])
    for seq, event in enumerate(events):
        event["seq"] = seq
    events.append(envelope(len(events), "cancelled", legs=[
        {"stage": "image", "slot_id": "slot-0", "slot_index": 0,
         "state": "aborted"},
        {"stage": "video", "slot_id": "slot-1", "slot_index": 1,
         "state": "completed"},
        {"stage": "image", "slot_id": "slot-2", "slot_index": 2,
         "state": "failed", "error": "render denied"},
        {"stage": "tts", "state": "aborted"},
    ]))
    return events


def sse_bytes(events):
    frames = []
    for unified in events:
        text = unified.get("text", "") if unified["kind"] == "vlm_delta" else ""
        chunk = {
            "id": "http-narrative-1", "object": "chat.completion.chunk",
            "created": 1764000000, "model": "narrative-fixture",
            "choices": [{"index": 0, "delta": {"content": text},
                         "finish_reason": None}],
            "unified": unified,
        }
        frames.append(b"data: " + json.dumps(
            chunk, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False).encode("utf-8") + b"\n\n")
    frames.append(b"data: [DONE]\n\n")
    return b"".join(frames)
