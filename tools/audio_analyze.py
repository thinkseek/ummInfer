#!/usr/bin/env python3
"""Analyze a PCM16 WAV for the defects a listening test notices.

Reports the waveform in fixed windows and flags five things a listener hears:
a buzz (a run of near-Nyquist windows inside the speech), tail_noise (the same
near-Nyquist noise at the very end of the file), a click (a sample step far
larger than its neighbours), a silent gap, and clipping. Exit code 1 when any
finding is present, so this can gate a generated artifact.

    python3 -I tools/audio_analyze.py out.wav
    python3 -I tools/audio_analyze.py out.wav --json
"""
import argparse
import json
import math
import struct
import sys
import wave

NOISE_WINDOW = 0.01     # 0.2 s of decoder noise is invisible in a 0.5 s window
NOISE_ZCR_RATIO = 0.4   # near Nyquist: 0.4 * sample_rate crossings per second
NOISE_SILENCE_RMS = 60.0  # same window and floor as native/tts/tail_trim.cpp
BUZZ_MIN_SECONDS = 0.2  # real fricative bursts measured up to 0.16 s
TAIL_NOISE_MIN_SECONDS = 0.1
CLICK_JUMP = 24000      # PCM16 sample-to-sample step
CLICK_MIN_SECONDS = None
SILENCE_RMS = 50.0
SILENCE_MIN_SECONDS = 0.25
CLIP_LEVEL = 32000


def read_wav(path):
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit("only PCM16 WAV is supported")
        frames = w.getnframes()
        rate = w.getframerate()
        channels = w.getnchannels()
        raw = w.readframes(frames)
    samples = struct.unpack("<%dh" % (len(raw) // 2), raw)
    if channels > 1:
        samples = samples[::channels]
    return list(samples), rate


def windows(samples, rate, seconds):
    step = max(1, int(rate * seconds))
    for start in range(0, len(samples), step):
        yield start, samples[start:start + step]


def rms_of(block):
    if not block:
        return 0.0
    return math.sqrt(sum(float(x) * x for x in block) / len(block))


def zcr_of(block, rate):
    if len(block) < 2:
        return 0.0
    crossings = sum(1 for i in range(1, len(block))
                    if (block[i - 1] < 0) != (block[i] < 0))
    return crossings / (len(block) / rate)


def periodicity(block, rate, max_hz=2500):
    """Normalised autocorrelation peak over lags that a buzz would repeat at.

    Speech, including fricatives, drifts in frequency and does not correlate
    with itself across a full window. A ringing codec repeats one period, so
    the peak approaches 1.
    """
    if len(block) < 64:
        return 0.0
    step = max(1, rate // 6000)          # decimate to keep this cheap
    data = block[::step]
    rate = rate // step
    lowest_lag = max(2, rate // max_hz)
    highest_lag = min(len(data) // 2, rate // 80)
    energy = sum(float(x) * x for x in data)
    if energy <= 0:
        return 0.0
    best = 0.0
    for lag in range(lowest_lag, highest_lag + 1):
        acc = 0.0
        for i in range(len(data) - lag):
            acc += float(data[i]) * data[i + lag]
        best = max(best, acc / energy)
    return best


def noise_kind(row, rate):
    """Classify one short window as silence, near-Nyquist noise, or speech."""
    if row["rms"] < NOISE_SILENCE_RMS:
        return "silence"
    return "noise" if row["zcr"] >= NOISE_ZCR_RATIO * rate else "voice"


def noise_windows(samples, rate):
    rows = []
    for start, block in windows(samples, rate, NOISE_WINDOW):
        rows.append({
            "start_s": start / rate,
            "end_s": min(len(samples), start + int(rate * NOISE_WINDOW)) / rate,
            "rms": rms_of(block),
            "zcr": zcr_of(block, rate),
        })
    return rows


def noise_duration(rows, rate):
    return sum(row["end_s"] - row["start_s"] for row in rows
               if noise_kind(row, rate) == "noise")


def noise_runs(rows, rate, until_s=None):
    """Maximal spans of noise or silence before until_s; a voice window ends one."""
    run = []
    for row in rows:
        if until_s is not None and row["start_s"] >= until_s:
            break
        if noise_kind(row, rate) == "voice":
            if run:
                yield run
            run = []
        else:
            run.append(row)
    if run:
        yield run


def tail_noise_run(rows, rate):
    """Walk backwards over the trailing run of near-Nyquist noise.

    Trailing and interleaved silent windows are skipped. Returns
    (start_s, end_s, noise_seconds) or None when the file ends on speech.
    """
    start = end = None
    noise = 0.0
    for row in reversed(rows):
        kind = noise_kind(row, rate)
        if kind == "voice":
            break
        if kind == "noise":
            noise += row["end_s"] - row["start_s"]
            start = row["start_s"]
            if end is None:
                end = row["end_s"]
    return None if start is None else (start, end, noise)


def analyze(samples, rate, window=0.5):
    rows = []
    for start, block in windows(samples, rate, window):
        rows.append({
            "start_s": round(start / rate, 2),
            "rms": round(rms_of(block), 1),
            "zcr": round(zcr_of(block, rate)),
            "periodicity": round(periodicity(block, rate), 3),
        })
    findings = []

    # Decoder noise lasts 0.2-0.5 s, so a 0.5 s window dilutes it below any
    # threshold; a short-window pass sees it. Zero-crossing rate alone cannot
    # separate noise from a fricative (both reach 0.4 * rate for up to 0.16 s);
    # duration and position do. The trailing run is reported as tail_noise, so
    # the interior scan stops where that run starts.
    short = noise_windows(samples, rate)
    tail = tail_noise_run(short, rate)
    if tail and tail[2] >= TAIL_NOISE_MIN_SECONDS:
        findings.append({
            "kind": "tail_noise",
            "start_s": round(tail[0], 2),
            "end_s": round(tail[1], 2),
            "detail": "%.0f ms of zcr >= %d" % (
                tail[2] * 1000, NOISE_ZCR_RATIO * rate),
        })

    for run in noise_runs(short, rate, tail[0] if tail else None):
        noise = noise_duration(run, rate)
        if noise < BUZZ_MIN_SECONDS:
            continue
        noisy = [row for row in run if noise_kind(row, rate) == "noise"]
        findings.append({
            "kind": "buzz",
            "start_s": round(noisy[0]["start_s"], 2),
            "end_s": round(noisy[-1]["end_s"], 2),
            "detail": "%.0f ms of zcr %d-%d, rms %d-%d" % (
                noise * 1000, int(noisy[0]["zcr"]), int(noisy[-1]["zcr"]),
                int(noisy[0]["rms"]), int(noisy[-1]["rms"])),
        })

    # Click: a single step far above anything its neighbours produce.
    clicks = []
    for i in range(1, len(samples)):
        step = abs(samples[i] - samples[i - 1])
        if step >= CLICK_JUMP:
            clicks.append(i)
    if clicks:
        findings.append({
            "kind": "click",
            "count": len(clicks),
            "first_s": round(clicks[0] / rate, 3),
            "detail": "max step %d" % max(
                abs(samples[i] - samples[i - 1]) for i in clicks[:64]),
        })

    # Silence inside the speech, not at either end.
    voiced = [i for i, row in enumerate(rows) if row["rms"] >= SILENCE_RMS]
    if voiced:
        first, last = voiced[0], voiced[-1]
        run = []
        for index in range(first, last + 1):
            if rows[index]["rms"] < SILENCE_RMS:
                run.append(index)
            else:
                if len(run) * window >= SILENCE_MIN_SECONDS:
                    findings.append({
                        "kind": "gap",
                        "start_s": rows[run[0]]["start_s"],
                        "end_s": round(rows[run[-1]]["start_s"] + window, 2),
                        "detail": "rms below %d" % SILENCE_RMS,
                    })
                run = []

    clipped = sum(1 for x in samples if abs(x) >= CLIP_LEVEL)
    if clipped:
        findings.append({"kind": "clipping", "samples": clipped})

    return {
        "duration_s": round(len(samples) / rate, 2),
        "sample_rate": rate,
        "samples": len(samples),
        "peak": max((abs(x) for x in samples), default=0),
        "rms": round(rms_of(samples), 1),
        "windows": rows,
        "findings": findings,
        "passed": not findings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--window", type=float, default=0.5)
    args = parser.parse_args()

    samples, rate = read_wav(args.wav)
    report = analyze(samples, rate, args.window)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("%s  %.2fs  %d Hz  peak %d  rms %.1f" % (
            args.wav, report["duration_s"], rate, report["peak"], report["rms"]))
        for finding in report["findings"]:
            if "start_s" in finding:
                print("  %-10s %5.2f-%5.2fs  %s" % (
                    finding["kind"], finding["start_s"],
                    finding["end_s"], finding["detail"]))
            else:
                print("  %-10s %s" % (finding["kind"],
                                      {k: v for k, v in finding.items()
                                       if k != "kind"}))
        print("  PASS" if report["passed"] else "  FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
