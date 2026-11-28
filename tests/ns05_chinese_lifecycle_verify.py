#!/usr/bin/env python3
"""Independent raw Task #81 Chinese 43-leg lifecycle control for Task #88."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import wave

EXPECTED_TEXT = "883802d889885bd032b18a3dc21c8f7e1ce77773f020ab0ee3e69a816ac8b572"
EXPECTED_PCM = "2ce2c53454349fd70edceab97e6afe6d93cf8ffc22e7bcdf497c303222d94f78"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ordinal(path):
    return int(path.name.split("-")[2] if path.name.startswith("tts-output-")
               else path.name.split("-")[1])


def verify(root):
    root = Path(root).resolve()
    rows = [json.loads(line) for line in (root / "capture/trajectories.jsonl").read_text().splitlines()]
    require(len(rows) == 1, "one Chinese trajectory")
    record = rows[0]
    require(record["terminal_state"] == "completed" and
            record["clean_text_sha256"] == EXPECTED_TEXT and
            sha(record["clean_text"].encode()) == EXPECTED_TEXT, "frozen Chinese clean text")
    public = [item for item in record["artifacts"] if item["media_type"] == "audio"]
    require(len(public) == 537, "537 public PCM chunks")
    import msgpack
    submits, outputs = [], []
    for path in sorted(root.glob("*.msgpack"), key=ordinal):
        message = msgpack.unpackb(path.read_bytes(), raw=False)
        if path.name.startswith("tts-output-") and message.get("type") in ("stream", "complete"):
            outputs.append(message)
        elif path.name.startswith("tts-") and message.get("type") == "submit":
            submits.append(message)
    require(len(submits) == 43, "43 native TTS submits")
    cursor = 0
    aggregate = []
    joins = []
    for submit in sorted(submits, key=lambda row: int(row["request_id"].rsplit(":", 1)[1])):
        child = submit["request_id"]
        native = [message for message in outputs if message.get("type") == "stream"
                  and message.get("request_id") == child]
        terminals = [message for message in outputs if message.get("type") == "complete"
                     and message.get("request_id") == child]
        require(native and len(terminals) == 1 and terminals[0].get("success") is True,
                f"native terminal: {child}")
        leg = []
        for index, message in enumerate(native):
            chunk = message["chunk"]
            require(chunk["index"] == index and cursor < len(public), f"chunk index: {child}")
            item = public[cursor]
            require(item["stream_index"] == index and
                    (root / "capture/artifacts" / item["ref"]).read_bytes() == chunk["audio"],
                    f"public/native PCM: {child}:{index}")
            leg.append(chunk["audio"])
            cursor += 1
        with wave.open(io.BytesIO(terminals[0]["result"]["data"]), "rb") as wav:
            require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 24000),
                    "WAV format")
            terminal = wav.readframes(wav.getnframes())
        require(b"".join(leg) == terminal, f"native stream/terminal: {child}")
        aggregate.append(terminal)
        joins.append({"request_id": child, "chunks": len(leg), "bytes": len(terminal),
                      "pcm_sha256": sha(terminal)})
    require(cursor == len(public), "unowned public PCM")
    pcm = b"".join(aggregate)
    require(len(pcm) == 14987520 and sha(pcm) == EXPECTED_PCM, "frozen aggregate PCM")
    log = (root / "tts.log").read_text(errors="replace")
    require(log.count("natural_eos=1") == 43 and log.count("tts request request_id=") == 43,
            "43/43 natural EOS")
    traces = [json.loads(line) for line in (root / "server.trace.jsonl").read_text().splitlines()]
    require(any(row.get("event") == "request_end" and row.get("status") == "completed"
                for row in traces), "public completed terminal")
    if (root / "gpu-monitor.json").exists():
        gpu = json.loads((root / "gpu-monitor.json").read_text())
        final_memory = gpu[-1]["memory_used_mib"] if gpu else 999999
    elif (root / "cleanup.json").exists():
        final_memory = json.loads((root / "cleanup.json").read_text())["post"]["memory_used_mib"]
    else:
        final_memory = None  # Accepted Task #81 raw-only fixture has no separate GPU file.
    require(final_memory is None or final_memory <= 16, "GPU cleanup")
    return {"status": "PASS", "natural_eos": "43/43", "native_terminals": 43,
        "public_pcm_chunks": cursor, "public_pcm_bytes": len(pcm),
        "clean_text_sha256": EXPECTED_TEXT, "aggregate_pcm_sha256": sha(pcm),
        "per_leg": joins, "generic_no_video_postprocessor": "NOT_USED",
        "gpu_cleanup": "PASS" if final_memory is not None else "NOT_PRESENT_IN_ACCEPTED_RAW_FIXTURE",
        "physical_acoustic_and_perceptual": "NOT_RUN"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = verify(args.root)
        Path(args.output).write_bytes(json.dumps(result, indent=2, sort_keys=True,
            ensure_ascii=False).encode() + b"\n")
        print(json.dumps({"status": "PASS", "natural_eos": "43/43"}, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"CHINESE CONTROL VERIFY FAIL: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
