#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent raw verifier for the preregistered NS-06/NS-08 paired gate.

Imports no live driver, trajectory producer, or scoring implementation. Only the
previously independent raw-media helpers are reused. Every reference, native byte,
terminal, PCM/WAV join, decoded image/video, and production-client lifecycle output
is rebuilt from retained evidence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
sys.path.insert(0, str(TESTS))
import cross_slot_visual_continuity_verify as raw  # noqa: E402

FORMAT = "umminfer-ns06-paired-76-v1"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def load_prereg(expected_sha):
    path = ROOT / "benchmarks/cross-slot-visual-continuity/ns06-preregistration.json"
    payload = path.read_bytes()
    require(sha_bytes(payload) == expected_sha, "preregistration digest mismatch")
    return json.loads(payload)


def read_capture(root, arm, kind):
    """Read <arm>/<kind> without confusing raw helper's directory argument."""
    base = Path(root) / arm
    record_path = base / kind / "trajectories.jsonl"
    artifact_dir = base / kind / "artifacts"
    # raw.read_capture expects capture names relative to the supplied root.
    # Use the arm as its root and keep the exact kind directory.
    return raw.read_capture(base, kind, {"record": f"{kind}/trajectories.jsonl",
                                        "artifact_dir": f"{kind}/artifacts"})


def visual_legs(record):
    return [leg for leg in record["terminal_legs"] if leg.get("slot_id") is not None]


def public_by_slot(capture):
    out = {}
    for item in capture["record"]["artifacts"]:
        slot = item.get("slot_index")
        if slot is not None:
            require(slot not in out, f"duplicate public artifact for slot {slot}")
            out[slot] = item
    return out


def native_submits(root, kind):
    import msgpack
    out = {}
    for path in sorted(root.glob(f"{kind}-*.msgpack")):
        message = msgpack.unpackb(path.read_bytes(), raw=False)
        if message.get("type") != "submit":
            continue
        request_id = message.get("request_id", "")
        if ":slot:" not in request_id:
            continue
        slot = int(request_id.rsplit(":slot:", 1)[1])
        require(slot not in out, f"duplicate native {kind} submit for slot {slot}")
        out[slot] = (path, message)
    return out


def verify_references(prereg, arm, report, capture, out):
    record = capture["record"]
    slots = capture["slots"]
    require(record["terminal_state"] == "completed" and len(slots) == 6,
            f"{arm}: main did not complete with six visual slots")
    require([slot["slot_index"] for slot in slots] == list(range(6)),
            f"{arm}: nonmonotonic visual slots")
    require([slot["media_type"] for slot in slots]
            == [spec["kind"] for spec in prereg["slots"]], f"{arm}: modalities changed")
    states = [leg["state"] for leg in visual_legs(record)]
    expected = prereg["expected_outcomes"][arm]
    require(states == expected["visual_leg_states"],
            f"{arm}: frozen visual failure/success states differ: {states}")
    public = public_by_slot(capture)
    images = [item for item in public.values() if item["media_type"] == "image"]
    videos = [item for item in public.values() if item["media_type"] == "video"]
    require(len(images) == expected["keyframe_count"]
            and len(videos) == expected["i2v_count"],
            f"{arm}: public image/video counts differ")
    image_native = native_submits(out, "image")
    video_native = native_submits(out, "video")
    require(sorted(image_native) == ([0] if arm == "before" else [0, 2, 4])
            and sorted(video_native) == ([1] if arm == "before" else [1, 3, 5]),
            f"{arm}: native submit set differs from frozen reference capability")
    raw_by_slot = {}
    for slot_id, item in public.items():
        raw_by_slot[slot_id] = (capture["artifact_root"] / item["ref"]).read_bytes()
        require(sha_bytes(raw_by_slot[slot_id]) == item["sha256"],
                f"{arm}: CAS hash mismatch for {slot_id}")
    chain = []
    for spec in prereg["slots"]:
        index, source = spec["index"], spec["source"]
        if source is None:
            continue
        leg = next(leg for leg in visual_legs(record)
                   if leg.get("slot_index") == index)
        if arm == "before" and index != 1:
            require(leg["state"] == "failed" and index not in public
                    and index not in (image_native if spec["kind"] == "image" else video_native),
                    f"{arm}: failed reference slot {index} generated anyway")
            chain.append({"shot": spec["shot"], "dependent_slot": index,
                          "source_slot": source, "state": "failed", "native_submit": False})
            continue
        require(leg["state"] == "completed" and index in public,
                f"{arm}: resolved slot {index} is not completed")
        source_item = public[source]
        source_raw = raw_by_slot[source]
        digest = source_item["sha256"]
        native = image_native if spec["kind"] == "image" else video_native
        path, message = native[index]
        inputs = message["data"]["request"]["inputs"]
        require(inputs.get("image") == source_raw
                and inputs.get("anchor_sha256") == digest
                and inputs.get("anchor_slot_index") == source
                and inputs.get("anchor_explicit") is True
                and "source=" not in inputs.get("prompt", ""),
                f"{arm}: native slot {index} did not consume exact slot {source} bytes")
        if spec["kind"] == "image":
            require("anchor_crop" not in inputs, "image edit gained Wan crop metadata")
        else:
            require(inputs.get("anchor_crop") == "center_cover",
                    "native I2V lost center-cover")
        dependent = public[index]
        require(dependent.get("anchor_slot_index") == source
                and dependent.get("anchor_explicit") is True
                and dependent.get("anchor_sha256") == digest
                and leg.get("anchor_sha256") == digest,
                f"{arm}: public/terminal anchor digest mismatch for slot {index}")
        chain.append({"shot": spec["shot"], "dependent_slot": index,
                      "source_slot": source, "state": "completed", "native_submit": True,
                      "source_sha256": digest,
                      "native_submit_sha256": sha_bytes(path.read_bytes()),
                      "dependent_sha256": dependent["sha256"],
                      "native_binary_source_equal": True})
    require(len(chain) == 5, f"{arm}: reference controls missing")
    return chain


def verify_media(prereg, arm, capture, reconstruction, ffmpeg, ffprobe):
    results = []
    for item in capture["record"]["artifacts"]:
        kind = item["media_type"]
        if kind not in ("image", "video"):
            continue
        path = capture["artifact_root"] / item["ref"]
        name = f"slot-{item['slot_index']}.{kind == 'image' and 'png' or 'mp4'}"
        target = reconstruction / name
        target.write_bytes(path.read_bytes())
        width, height, payload = raw.rgb(ffmpeg, target)
        rgb_name = f"{name}.{width}x{height}.rgb"
        (reconstruction / rgb_name).write_bytes(payload)
        entry = {"slot_index": item["slot_index"], "media_type": kind,
                 "sha256": sha_bytes(path.read_bytes()), "decoded_dimensions": [width, height],
                 "rgb_sha256": sha_bytes(payload)}
        if kind == "image":
            require((width, height) == (512, 512), f"{arm}: keyframe dimensions changed")
        else:
            require((width, height) == (704, 480), f"{arm}: I2V dimensions changed")
            source_index = prereg["slots"][item["slot_index"]]["source"]
            source_item = next(x for x in capture["record"]["artifacts"]
                               if x.get("slot_index") == source_index)
            source_path = capture["artifact_root"] / source_item["ref"]
            _, _, source_eval = raw.rgb(ffmpeg, source_path, evaluation=True)
            _, _, first_eval = raw.rgb(ffmpeg, target, "select=eq(n\\,0)", True)
            _, _, last_eval = raw.rgb(ffmpeg, target, "select=eq(n\\,28)", True)
            first_mae = round(raw.mae(source_eval, first_eval), 6)
            motion_mae = round(raw.mae(first_eval, last_eval), 6)
            require(first_mae <= 48.0 and motion_mae >= 1.0,
                    f"{arm}: slot {item['slot_index']} RGB proxy gate failed")
            probe = subprocess.run([
                ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=codec_name,width,height,nb_frames,r_frame_rate,pix_fmt", "-of", "json",
                str(target)], check=True, capture_output=True, text=True, timeout=30)
            stream = json.loads(probe.stdout)["streams"][0]
            require(stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
                    and (stream["width"], stream["height"]) == (704, 480)
                    and int(stream["nb_frames"]) == 29 and stream["r_frame_rate"] == "16/1",
                    f"{arm}: invalid decoded I2V stream")
            entry.update({"source_slot": source_index, "first_frame_mae": first_mae,
                          "motion_mae": motion_mae, "video_stream": stream,
                          "first_eval_sha256": sha_bytes(first_eval),
                          "last_eval_sha256": sha_bytes(last_eval)})
        results.append(entry)
    expected = prereg["expected_outcomes"][arm]
    require(sum(x["media_type"] == "image" for x in results) == expected["keyframe_count"]
            and sum(x["media_type"] == "video" for x in results) == expected["i2v_count"],
            f"{arm}: decoded media counts differ")
    return results


def verify_tts(arm, report, capture):
    public = [item for item in capture["record"]["artifacts"] if item["media_type"] == "audio"]
    require(public and report["tts_submits"]
            and len(report["tts_legs"]) == len(report["tts_submits"]),
            f"{arm}: narration leg inputs/joins are incomplete")
    public_raw = [(item, (capture["artifact_root"] / item["ref"]).read_bytes()) for item in public]
    for item, pcm in public_raw:
        require(pcm and len(pcm) % 2 == 0, f"{arm}: invalid PCM16")
        samples = [int.from_bytes(pcm[i:i+2], "little", signed=True)
                   for i in range(0, len(pcm), 2)]
        require(math.sqrt(sum(x*x for x in samples) / len(samples)) > 1.0,
                f"{arm}: silent narration")
    # The independent raw helper intentionally has no producer join function. Join
    # every native chunk and terminal WAV directly here.
    cursor = 0
    verified = []
    for index, submit in enumerate(report["tts_submits"]):
        child = submit["request_id"]
        submit_message = raw.packed_message(report["_evidence_root"], submit)
        text = submit_message["data"]["request"]["inputs"]["text"]
        require(text == submit["text"], f"{arm}: TTS text/raw submit mismatch")
        chunks = [x for x in report["tts_output"] if x.get("message_type") == "stream"
                  and x.get("request_id") == child]
        terminals = [x for x in report["tts_output"] if x.get("message_type") == "complete"
                     and x.get("request_id") == child]
        require(chunks and len(terminals) == 1 and terminals[0].get("success") is True,
                f"{arm}: incomplete TTS leg {child}")
        native_parts = []
        for expected_index, chunk in enumerate(chunks):
            message = raw.packed_message(report["_evidence_root"], chunk)
            pcm = message["chunk"]["audio"]
            require(message["chunk"]["index"] == expected_index and cursor < len(public_raw)
                    and public_raw[cursor][1] == pcm
                    and public_raw[cursor][0].get("stream_index") == expected_index,
                    f"{arm}: public/native PCM mismatch at {child}:{expected_index}")
            native_parts.append(pcm)
            cursor += 1
        complete = raw.packed_message(report["_evidence_root"], terminals[0])
        import io
        import wave
        with wave.open(io.BytesIO(complete["result"]["data"]), "rb") as wav:
            require(wav.getnchannels() == 1 and wav.getsampwidth() == 2
                    and wav.getframerate() == 24000, f"{arm}: invalid terminal WAV")
            wav_pcm = wav.readframes(wav.getnframes())
        joined = b"".join(native_parts)
        require(joined == wav_pcm, f"{arm}: PCM/WAV mismatch for {child}")
        expected = report["tts_legs"][index]
        require(expected["request_id"] == child and expected["pcm_sha256"] == sha_bytes(joined)
                and expected["terminal_wav_sha256"] == sha_bytes(complete["result"]["data"]),
                f"{arm}: reported TTS join differs")
        verified.append({"request_id": child, "text_sha256": sha_bytes(text.encode()),
                         "pcm_sha256": sha_bytes(joined), "pcm_bytes": len(joined),
                         "terminal_wav_sha256": sha_bytes(complete["result"]["data"])})
    require(cursor == len(public_raw), f"{arm}: unowned public PCM chunks")
    return {"legs": verified, "native_leg_count": len(verified),
            "native_leg_target_met": len(verified) == 3,
            "pcm_sha256": sha_bytes(b"".join(x[1] for x in public_raw)),
            "pcm_bytes": sum(len(x[1]) for x in public_raw),
            "text_sha256": sha_bytes(" ".join(x["text"] for x in report["tts_submits"]).encode())}


def verify_client_and_reconnect(arm, report, main, reconnect, replay_parent, node, prereg):
    root = Path(report["_evidence_root"])
    out = root / "client" / "out"
    export_bytes = (out / "export.json").read_bytes()
    require(export_bytes == (out / "reexport.json").read_bytes(),
            f"{arm}: export/re-export bytes differ")
    exported = json.loads(export_bytes)
    lifecycle = report["client_lifecycle"]
    require(lifecycle["clear"] is True and lifecycle["exact_reexport"] is True
            and lifecycle["network_calls"] == 0
            and lifecycle["raw_media_and_pcm_restoration"]
                == "NOT_SUPPORTED_BY_EXISTING_SESSION_SCHEMA",
            f"{arm}: client lifecycle or media limitation changed")
    require(len(exported["turns"]) == 1
            and exported["turns"][0]["response"] == main["clean"]
            and exported["turns"][0]["prompt"]
                == json.loads((root / "request.json").read_text())["messages"][0]["content"][0]["text"],
            f"{arm}: exported text does not match real public request/response")
    exported_slots = exported["turns"][0]["visual_slots"]
    public = public_by_slot(main)
    require(len(exported_slots) == 6, f"{arm}: export lost visual slots")
    for spec, slot in zip(prereg["slots"], exported_slots):
        require(slot["id"] == f"slot-{spec['index']}" and slot["kind"] == spec["kind"],
                f"{arm}: export slot identity changed")
        if spec["source"] is not None and spec["index"] in public:
            require(slot.get("anchor_slot_index") == spec["source"]
                    and slot.get("anchor_explicit") is True,
                    f"{arm}: export resolved source reference changed")
        elif spec["source"] is not None:
            require(slot.get("error") and not slot.get("anchor_sha256"),
                    f"{arm}: unresolved export reference invented a digest or lost its error")
        if spec["index"] in public:
            require(slot.get("sha256") == public[spec["index"]]["sha256"]
                    and slot.get("bytes") == public[spec["index"]]["bytes"],
                    f"{arm}: export artifact identity changed")
            if spec["source"] is not None:
                require(slot.get("anchor_sha256") == public[spec["index"]].get("anchor_sha256"),
                        f"{arm}: export resolved source digest changed")
    reconnect_body = json.loads((out / "reconnect-body.json").read_bytes())
    require(reconnect_body["messages"][:-1] == [
        {"role": "user", "content": exported["turns"][0]["prompt"]},
        {"role": "assistant", "content": exported["turns"][0]["response"]}],
        f"{arm}: reconnect did not use imported history")
    require(reconnect_body["messages"][-1]
            == {"role": "user", "content": "Reply with exactly READY."}
            and reconnect_body["session_id"] == exported["session_id"]
            and reconnect_body["modalities"] == ["text"],
            f"{arm}: reconnect body changed")
    require(reconnect["record"]["terminal_state"] == "completed"
            and reconnect["clean"].strip() == "READY" and not reconnect["rebuilt"],
            f"{arm}: real reconnect after import failed")
    replay = replay_parent / f"{arm}-client"
    require(not replay.exists(), f"{arm}: client replay directory already exists")
    result = subprocess.run([
        node, str(root / "client" / "production-lifecycle.mjs"),
        str(root / "client" / "input-state.json"), str(replay)],
        capture_output=True, text=True, timeout=60)
    require(result.returncode == 0, f"{arm}: fresh production-client replay failed: {result.stderr}")
    for name in ("export.json", "export.md", "reexport.json", "reconnect-body.json",
                 "result.json", "clear.json"):
        require((replay / name).read_bytes() == (out / name).read_bytes(),
                f"{arm}: fresh client replay differs for {name}")
    return {"exact_reexport": True, "clear_import_reconnect": True,
            "network_calls_during_offline_lifecycle": 0,
            "raw_media_and_pcm_restoration": "NOT_SUPPORTED_BY_EXISTING_SCHEMA",
            "exported_slot_count": 6,
            "reconnect_text": reconnect["clean"].strip(),
            "fresh_client_replay": True}


def verify_arm(root, arm, prereg, reconstruction_parent, ffmpeg, ffprobe, node):
    report = json.loads((root / arm / "arm-report.json").read_text())
    report["_evidence_root"] = root / arm
    main = read_capture(root, arm, "main")
    reconnect = read_capture(root, arm, "reconnect")
    chain = verify_references(prereg, arm, report, main, root / arm)
    reconstruction = reconstruction_parent / arm
    reconstruction.mkdir()
    media = verify_media(prereg, arm, main, reconstruction, ffmpeg, ffprobe)
    tts = verify_tts(arm, report, main)
    normalized_clean = " ".join(main["clean"].split())
    require(normalized_clean == " ".join(prereg["expected_clean_prose_sentences"]),
            f"{arm}: clean prose differs from three frozen narration sentences")
    require(" ".join(x["text"] for x in report["tts_submits"]).split()
            == normalized_clean.split(), f"{arm}: TTS text differs from clean prose")
    asr = "".join(event.get("text", "") for _, event in main["events"]
                  if event["kind"] == "asr_final")
    require(asr, f"{arm}: real voice input produced no ASR final")
    client = verify_client_and_reconnect(arm, report, main, reconnect,
                                         reconstruction_parent, node, prereg)
    require(report["native_narration_legs"] == tts["native_leg_count"]
            and report["narration_leg_target_met"] == tts["native_leg_target_met"],
            f"{arm}: native narration target accounting mismatch")
    require(report["post_terminal_abort"] == {"status": 200, "success": True}
            and report["health_final"].get("in_flight") == 0
            and report["manager"]["images_settled_before_wan"] is True
            and report["manager"]["asr_settled_before_wan"] is True,
            f"{arm}: terminal/recovery/stage-switch contract failed")
    processes = report["stage_lifecycle"]["processes"]
    require(len(processes) == 7 and all(x["returncode"] == 0 for x in processes),
            f"{arm}: task-owned process cleanup failed")
    gpu = report["gpu7"]["after_cleanup"]
    require(gpu["index"] == 7 and gpu["memory_used_mib"] <= 16
            and gpu["utilization_percent"] == 0 and gpu.get("compute_processes") == 0,
            f"{arm}: GPU 7 was not cleaned")
    require(report["runtime_commit"] == prereg[f"{arm}_runtime_commit"],
            f"{arm}: runtime commit changed")
    return {"chain": chain, "media": media, "tts": tts, "client": client,
            "clean_text_sha256": sha_bytes(main["clean"].encode()),
            "asr_final_sha256": sha_bytes(asr.encode()),
            "raw_sse_sha256": sha_bytes(main["raw"]),
            "main_terminal": "complete", "reconnect_terminal": "complete",
            "visual_states": [x["state"] for x in visual_legs(main["record"])],
            "gpu7_cleanup": gpu}


def verify_provenance(root, arm, node):
    report = json.loads((root / arm / "arm-report.json").read_text())
    provenance = report["provenance"]
    for name, item in provenance["binaries"].items():
        actual = raw.file_sha(item["path"])
        require(actual["sha256"] == item["sha256"] and actual["bytes"] == item["bytes"],
                f"{arm}: binary {name} changed")
    for name in ("live_harness", "independent_verifier", "web_client", "flux_stage", "node"):
        item = provenance[name]
        actual = raw.file_sha(node if name == "node" else item["path"])
        if name == "independent_verifier":
            # A verifier-only correction is reported distinctly, never aliased.
            continue
        require(actual["sha256"] == item["sha256"], f"{arm}: source {name} changed")
    models = {name: raw.verify_model(identity) for name, identity in report["models"].items()}
    return {"binaries": "REHASHED_EXACT", "models": models}


def self_check():
    raw.self_check()
    print("OFFLINE_NS06_VERIFY: PASS (reuses independent raw/manifest primitives)")
    print("REAL_GENERATION: NOT_RUN")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, nargs="?")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--check-manifest", action="store_true")
    parser.add_argument("--skip-provenance-rehash", action="store_true")
    parser.add_argument("--reconstruction-dir", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--node", default="node")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    if args.evidence is None or args.out is None:
        parser.error("verification requires evidence and --out (or --self-check)")
    root = args.evidence.resolve()
    manifest = raw.verify_manifest(root) if args.check_manifest else {"checked": False}
    top = json.loads((root / "report.json").read_text())
    require(top["format"] == FORMAT and top["result"] == "PAIRED_TECHNICAL_PASS"
            and top["ns06_full_pass"] is False
            and top["ns06_human_character_style_threshold"] == "NOT_RUN / NOT_MET",
            "top-level claim boundary changed")
    prereg = load_prereg(top["preregistration_sha256"])
    reconstruction = args.reconstruction_dir or root / "reconstructed"
    require(not reconstruction.exists(), "reconstruction directory already exists")
    reconstruction.mkdir(parents=True)
    arms = {arm: verify_arm(root, arm, prereg, reconstruction,
                            args.ffmpeg, args.ffprobe, args.node)
            for arm in ("before", "after")}
    before, after = arms["before"], arms["after"]
    require(before["clean_text_sha256"] == after["clean_text_sha256"]
            and before["asr_final_sha256"] == after["asr_final_sha256"]
            and before["tts"]["pcm_sha256"] == after["tts"]["pcm_sha256"]
            and before["tts"]["text_sha256"] == after["tts"]["text_sha256"],
            "Before/After ASR/prose/PCM identity diverged")
    require(before["visual_states"] == prereg["expected_outcomes"]["before"]["visual_leg_states"]
            and after["visual_states"] == prereg["expected_outcomes"]["after"]["visual_leg_states"],
            "paired technical outcome changed")
    after_i2v = [x for x in after["chain"] if x["dependent_slot"] in (1, 3, 5)]
    require(len(after_i2v) == 3 and all(x["native_binary_source_equal"] for x in after_i2v)
            and len([x for x in after["chain"] if x["dependent_slot"] in (2, 4)]) == 2
            and len([x for x in before["chain"] if x.get("native_submit")]) == 1,
            "candidate did not bind all three I2V chains plus both edits")
    provenance = {"skipped": args.skip_provenance_rehash}
    if not args.skip_provenance_rehash:
        provenance = {"skipped": False,
                      "before": verify_provenance(root, "before", args.node),
                      "after": verify_provenance(root, "after", args.node)}
    result = {
        "result": "PASS_PAIRED_TECHNICAL",
        "technical_gain": {"before_keyframes_i2v": [1, 1], "after_keyframes_i2v": [3, 3],
                           "after_exact_edit_links": 2, "after_exact_i2v_links": 3,
                           "clean_prose_asr_pcm_equal": True},
        "before": before, "after": after,
        "ns06_human_character_style_threshold": "NOT_RUN / NOT_MET",
        "ns06_full_pass": False,
        "ns06_narration_leg_target": {
            "expected_native_legs": 3,
            "before_actual": before["tts"]["native_leg_count"],
            "after_actual": after["tts"]["native_leg_count"],
            "met": before["tts"]["native_leg_target_met"]
                and after["tts"]["native_leg_target_met"],
        },
        "ns08_no_regression": True,
        "ns08_raw_media_audio_restoration": "NOT_SUPPORTED_BY_EXISTING_SESSION_SCHEMA",
        "manifest": manifest, "provenance": provenance,
        "perceptual_quality": "NOT_RUN", "generalization": "NOT_COVERED",
        "training_optimizer": "NOT_RUN", "resident_service": "NOT_CONTACTED",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": result["result"], "technical_gain": result["technical_gain"],
                      "manifest": manifest, "ns06_full_pass": False,
                      "ns08_no_regression": True}, sort_keys=True))
    print("INDEPENDENT_NS06_NS08_PAIRED: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
