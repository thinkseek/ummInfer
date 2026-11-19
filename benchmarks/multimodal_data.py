# SPDX-License-Identifier: Apache-2.0
"""Resolve the versioned multimodal benchmark manifest into real input bytes.

Every input is public and identified by URL plus SHA-256. Nothing large is
committed. The cache is content-checked on every use: a cached file is reused
only when its size and SHA-256 match the manifest. Downloads are streamed to a
``.part`` file, bounded by the declared byte count, verified, then renamed.

All loops here have an explicit cap: bytes read, archive members scanned, and
line index for text rows. The flac/ffmpeg decode helpers run with a timeout.
"""

import array
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
import wave

SCHEMA_VERSION = "umm-multimodal-manifest/v1"
CHUNK = 1 << 20
MAX_DOWNLOAD_BYTES = 512 << 20
MAX_MANIFEST_BYTES = 4 << 20
MAX_JSON_BYTES = 4 << 20
MAX_MEMBER_BYTES = 64 << 20
MAX_TAR_MEMBERS = 200000
MAX_TEXT_LINE_BYTES = 64 << 10
MAX_RIFF_CHUNKS = 64
DECODE_TIMEOUT_S = 120
USER_AGENT = "umminfer-multimodal-benchmark/1"

AUDIO_FORMAT = (1, 2, 16000)  # PCM, mono, 16 kHz -- the worker contract
FLEURS_FORMAT = (3, 1, 16000, 32)  # IEEE float, mono, 16 kHz, 32-bit


class DataError(RuntimeError):
    """A pinned input is missing, corrupt, or does not match the manifest."""


# ---- pure helpers ----------------------------------------------------------

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, cap=MAX_DOWNLOAD_BYTES):
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(CHUNK)
            if not chunk:
                break
            size += len(chunk)
            if size > cap:
                raise DataError(f"{path}: larger than {cap} bytes")
            digest.update(chunk)
    return digest.hexdigest(), size


def load_manifest(path=None):
    path = Path(path) if path else (
        Path(__file__).resolve().with_name("multimodal_manifest.json"))
    raw = path.read_bytes()
    if len(raw) > MAX_MANIFEST_BYTES:
        raise DataError(f"{path}: manifest is too large")
    manifest = json.loads(raw)
    errors = validate_manifest(manifest)
    if errors:
        raise DataError(f"{path}: " + "; ".join(errors))
    return manifest


TASKS = ("asr", "gsm8k", "vqa", "joint")
METRICS = ("wer", "cer", "vqa_accuracy", "gsm8k_exact",
           "joint_asr_wer_keyword_hit")
INPUT_FIELDS = {
    "librispeech_utterance": ("member", "flac_sha256", "wav_sha256"),
    "fleurs_utterance": ("member", "sha256", "wav_sha256", "wav_bytes"),
    "dataset_row": ("row_index", "row_sha256"),
    "image_url": ("url", "sha256", "bytes"),
    "speechcoco_row": ("row_index", "revision", "audio_sha256",
                       "image_sha256"),
}


def validate_manifest(manifest):
    """Return a list of structural errors. An empty list means valid."""
    errors = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version != {SCHEMA_VERSION}")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        return errors + ["cases must be a nonempty list"]
    if manifest.get("total_cases") != len(cases):
        errors.append("total_cases does not match len(cases)")
    seen = set()
    for case in cases:
        cid = case.get("id")
        if not cid:
            errors.append("a case has no id")
            continue
        if cid in seen:
            errors.append(f"duplicate case id: {cid}")
        seen.add(cid)
        if case.get("task") not in TASKS:
            errors.append(f"{cid}: unknown task")
        if case.get("metric") not in METRICS:
            errors.append(f"{cid}: unknown metric")
        dataset = case.get("dataset")
        if dataset not in manifest.get("datasets", {}):
            errors.append(f"{cid}: unknown dataset {dataset}")
        kind = (case.get("input") or {}).get("type")
        if kind not in INPUT_FIELDS:
            errors.append(f"{cid}: unknown input type {kind}")
            continue
        for field in INPUT_FIELDS[kind]:
            if field not in case["input"]:
                errors.append(f"{cid}: input.{field} is required")
        if not case.get("reference"):
            errors.append(f"{cid}: reference is required")
        if case.get("task") == "vqa":
            answers = case.get("reference", {}).get("answers") or []
            if len(answers) != 10:
                errors.append(f"{cid}: VQA accuracy needs 10 answers")
        if case.get("task") == "gsm8k":
            if not isinstance(case.get("reference", {}).get("answer"), int):
                errors.append(f"{cid}: gsm8k reference.answer must be an integer")
    for name, dataset in manifest.get("datasets", {}).items():
        if not dataset.get("url"):
            errors.append(f"{name}: dataset url is required")
        if not dataset.get("license"):
            errors.append(f"{name}: dataset license is required")
        if not any(key in dataset for key in
                   ("sha256", "archive_sha256", "questions_sha256",
                    "revision")):
            errors.append(f"{name}: a dataset SHA-256 or revision is required")
    return errors


def coverage(manifest):
    """Counts per task and modality, plus the declared metric for each task."""
    tasks = {}
    for case in manifest["cases"]:
        row = tasks.setdefault(case["task"], {
            "cases": 0, "languages": set(), "modalities": set(),
            "metrics": set()})
        row["cases"] += 1
        row["languages"].add(case.get("language", "unknown"))
        row["modalities"].update(case.get("modalities", []))
        row["metrics"].add(case["metric"])
    return {name: {
        "cases": row["cases"],
        "languages": sorted(row["languages"]),
        "modalities": sorted(row["modalities"]),
        "metrics": sorted(row["metrics"]),
    } for name, row in sorted(tasks.items())}


# ---- download and cache ----------------------------------------------------

def _join(url, name):
    return url.rstrip("/") + "/" + urllib.parse.quote(name)


def download_verified(url, dest, expected_sha256, expected_bytes, *,
                      max_bytes=MAX_DOWNLOAD_BYTES, timeout=120):
    """Return `dest`, downloading it only when the cached bytes are wrong.

    The reader stops as soon as the declared byte count is exceeded, so an
    upstream that grew cannot fill the disk. A failed verification leaves no
    file at `dest`.
    """
    dest = Path(dest)
    expected_bytes = int(expected_bytes)
    if expected_bytes <= 0 or expected_bytes > max_bytes:
        raise DataError(f"{url}: declared size {expected_bytes} is outside "
                        f"1..{max_bytes}")
    if dest.exists():
        digest, size = sha256_file(dest, cap=max_bytes)
        if size == expected_bytes and digest == expected_sha256:
            return dest
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    total = 0
    max_reads = expected_bytes // CHUNK + 4
    reads = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with open(part, "wb") as out:
                while reads < max_reads:
                    reads += 1
                    chunk = response.read(min(CHUNK, expected_bytes + 1 - total))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > expected_bytes:
                        raise DataError(
                            f"{url}: upstream is larger than the declared "
                            f"{expected_bytes} bytes")
                    out.write(chunk)
        if total != expected_bytes:
            raise DataError(f"{url}: got {total} bytes, expected {expected_bytes}")
        digest, size = sha256_file(part, cap=max_bytes)
        if digest != expected_sha256:
            raise DataError(f"{url}: SHA-256 mismatch")
        os.replace(part, dest)
    finally:
        if part.exists():
            part.unlink()
    return dest


def fetch_json(url, max_bytes=MAX_JSON_BYTES, timeout=120):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise DataError(f"{url}: JSON response is too large")
    return json.loads(raw)


# ---- LibriSpeech -----------------------------------------------------------

def ensure_librispeech(manifest, cache_dir):
    dataset = manifest["datasets"]["librispeech_test_clean"]
    return download_verified(
        dataset["url"], Path(cache_dir) / "librispeech_test_clean/test-clean.tar.gz",
        dataset["archive_sha256"], dataset["archive_bytes"])


def ensure_fleurs_zh(manifest, cache_dir):
    dataset = manifest["datasets"]["fleurs_zh_cn_test"]
    return download_verified(
        dataset["url"], Path(cache_dir) / "fleurs_zh_cn_test/test.tar.gz",
        dataset["archive_sha256"], dataset["archive_bytes"])


def extract_members(archive_path, members, max_bytes=MAX_MEMBER_BYTES):
    """Scan the archive once and return {member: bytes}. Missing members fail."""
    wanted = set(members)
    found = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for count, info in enumerate(archive):
            if count >= MAX_TAR_MEMBERS:
                raise DataError(f"{archive_path}: more than {MAX_TAR_MEMBERS} members")
            if not info.isfile() or info.name not in wanted:
                continue
            if info.size > max_bytes:
                raise DataError(f"{info.name}: member is too large")
            handle = archive.extractfile(info)
            if handle is None:
                raise DataError(f"{info.name}: cannot be read")
            data = handle.read(max_bytes + 1)
            if len(data) != info.size:
                raise DataError(f"{info.name}: member is truncated")
            found[info.name] = data
            if len(found) == len(wanted):
                break
    missing = wanted - set(found)
    if missing:
        raise DataError("archive members are missing: " + ", ".join(sorted(missing)))
    return found


def decode_audio_to_wav(flac_bytes, expected_wav_sha256=None):
    """Decode FLAC to the single WAV format the workers accept.

    `flac` is preferred; `ffmpeg` is the fallback. Both decode losslessly, so
    the rebuilt canonical WAV is identical whichever tool ran.
    """
    raw = None
    errors = []
    if shutil.which("flac"):
        try:
            result = subprocess.run(
                ["flac", "-d", "-c", "--force-raw-format", "--endian=little",
                 "--sign=signed", "-"],
                input=flac_bytes, capture_output=True,
                timeout=DECODE_TIMEOUT_S)
            if result.returncode == 0 and result.stdout:
                raw = result.stdout
            else:
                errors.append("flac exited " + str(result.returncode))
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(f"flac: {error}")
    if raw is None:
        if not shutil.which("ffmpeg"):
            raise DataError("cannot decode FLAC: neither flac nor ffmpeg is on PATH")
        try:
            result = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "s16le",
                 "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", "pipe:1"],
                input=flac_bytes, capture_output=True,
                timeout=DECODE_TIMEOUT_S)
            if result.returncode == 0 and result.stdout:
                raw = result.stdout
            else:
                errors.append("ffmpeg exited " + str(result.returncode))
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(f"ffmpeg: {error}")
    if raw is None:
        raise DataError("FLAC decode failed: " + "; ".join(errors))
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(raw)
    data = out.getvalue()
    if expected_wav_sha256 and sha256_bytes(data) != expected_wav_sha256:
        raise DataError("decoded WAV does not match the manifest SHA-256")
    return data


def float_wav_to_pcm16(data, expected_wav_sha256=None):
    """Convert a mono 16 kHz IEEE-float WAV to the worker's PCM16 WAV.

    FLEURS stores 32-bit float samples. The float-to-int16 rule is fixed here
    (scale by 32768, truncate toward zero, clamp), so the output bytes do not
    depend on the ffmpeg build or on its dithering. Parsing is bounded by a
    chunk cap and by the file size; nothing is written to disk.
    """
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise DataError("audio is not a RIFF/WAVE file")
    fmt = frames = None
    offset = 12
    for _ in range(MAX_RIFF_CHUNKS):
        if offset + 8 > len(data):
            break
        chunk_id = data[offset:offset + 4]
        size = struct.unpack("<I", data[offset + 4:offset + 8])[0]
        body = offset + 8
        if body + size > len(data):
            raise DataError(f"RIFF chunk {chunk_id!r} exceeds the file")
        if chunk_id == b"fmt ":
            if size < 16:
                raise DataError("WAV fmt chunk is shorter than 16 bytes")
            tag, channels, rate, _, _, bits = struct.unpack(
                "<HHIIHH", data[body:body + 16])
            fmt = (tag, channels, rate, bits)
        elif chunk_id == b"data":
            frames = data[body:body + size]
            if fmt is not None:
                break
        offset = body + size + (size & 1)
    if fmt is None or frames is None:
        raise DataError("WAV is missing the fmt or data chunk")
    if fmt != FLEURS_FORMAT:
        raise DataError(f"audio must be float32 mono 16 kHz, got {fmt}")
    if not frames:
        raise DataError("audio has no frames")
    samples = array.array("f")
    samples.frombytes(frames[:len(frames) - len(frames) % 4])
    if sys.byteorder != "little":
        samples.byteswap()
    pcm = array.array("h", (
        -32768 if value <= -1.0 else
        32767 if value >= 1.0 else int(value * 32768.0)
        for value in samples))
    if sys.byteorder != "little":
        pcm.byteswap()
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm.tobytes())
    data = out.getvalue()
    if expected_wav_sha256 and sha256_bytes(data) != expected_wav_sha256:
        raise DataError("converted WAV does not match the manifest SHA-256")
    return data


def canonical_wav_bytes(data):
    """Validate PCM16/16 kHz/mono and rebuild a header-only WAV."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            got = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as error:
        raise DataError(f"audio is not a valid WAV: {error}")
    if got != AUDIO_FORMAT:
        raise DataError(f"audio must be PCM16 mono 16 kHz, got {got}")
    if not frames:
        raise DataError("audio has no frames")
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(frames)
    return out.getvalue()


# ---- per-case resolution ---------------------------------------------------

def _gsm8k_row(manifest, cache_dir, case):
    dataset = manifest["datasets"]["gsm8k_test"]
    path = download_verified(
        dataset["url"], Path(cache_dir) / "gsm8k_test/test.jsonl",
        dataset["sha256"], dataset["bytes"])
    wanted = case["input"]["row_index"]
    with open(path, "rb") as handle:
        for index, line in enumerate(handle):
            if index > wanted:
                break
            if index != wanted:
                continue
            line = line.rstrip(b"\n")
            if len(line) > MAX_TEXT_LINE_BYTES:
                raise DataError(f"{path}: row {wanted} is too long")
            if sha256_bytes(line) != case["input"]["row_sha256"]:
                raise DataError(f"{path}: row {wanted} does not match the manifest")
            row = json.loads(line)
            gold = int(row["answer"].split("####", 1)[1].strip().replace(",", ""))
            if gold != case["reference"]["answer"]:
                raise DataError(f"{path}: row {wanted} gold answer changed")
            return row["question"]
    raise DataError(f"{path}: row {wanted} is missing")


def _vqa_image(manifest, cache_dir, case):
    source = case["input"]
    name = f"{source['sha256']}.img"
    path = download_verified(
        source["url"], Path(cache_dir) / "vqav2_val/images" / name,
        source["sha256"], source["bytes"])
    data = path.read_bytes()
    if not (data.startswith(b"\xff\xd8") or
            data.startswith(b"\x89PNG\r\n\x1a\n")):
        raise DataError(f"{path}: not a JPEG or PNG image")
    return data


def _speechcoco(manifest, cache_dir, case):
    dataset = manifest["datasets"]["speechcoco_val"]
    source = case["input"]
    revision = source["revision"]
    if revision != dataset["revision"]:
        raise DataError(f"{case['id']}: revision does not match the dataset pin")
    audio_dest = (Path(cache_dir) / "speechcoco_val" /
                  f"row-{source['row_index']:04d}.wav")
    image_dest = (Path(cache_dir) / "speechcoco_val" /
                  f"row-{source['row_index']:04d}.jpg")
    # Warm-cache fast path: the manifest pins the revision URLs and byte
    # hashes, so verified local files need no live rows-API metadata call.
    # Any miss or mismatch falls through to the rows-API path below.
    try:
        audio_path = download_verified(
            source["audio_source"], audio_dest,
            source["audio_sha256"], source["audio_bytes"])
        image_path = download_verified(
            source["image_source"], image_dest,
            source["image_sha256"], source["image_bytes"])
        audio = canonical_wav_bytes(audio_path.read_bytes())
        image = image_path.read_bytes()
        if image.startswith(b"\xff\xd8"):
            return audio, image
    except (DataError, OSError):
        pass
    url = (f"{dataset['rows_api']}&revision={urllib.parse.quote(revision)}"
           f"&offset={int(source['row_index'])}&length=1")
    payload = fetch_json(url)["rows"]
    if len(payload) != 1:
        raise DataError(f"{case['id']}: rows API returned {len(payload)} rows")
    row = payload[0]["row"]
    if row.get("text") != case["reference"]["caption"]:
        raise DataError(f"{case['id']}: caption changed upstream")
    audio_url = row["audio"][0]["src"]
    image_url = row["image"]["src"]
    for name, link in (("audio", audio_url), ("image", image_url)):
        if f"/{revision}/" not in link:
            raise DataError(f"{case['id']}: {name} URL is not at the pinned revision")
    audio_path = download_verified(
        audio_url, Path(cache_dir) / "speechcoco_val" /
        f"row-{source['row_index']:04d}.wav",
        source["audio_sha256"], source["audio_bytes"])
    image_path = download_verified(
        image_url, Path(cache_dir) / "speechcoco_val" /
        f"row-{source['row_index']:04d}.jpg",
        source["image_sha256"], source["image_bytes"])
    audio = canonical_wav_bytes(audio_path.read_bytes())
    image = image_path.read_bytes()
    if not image.startswith(b"\xff\xd8"):
        raise DataError(f"{case['id']}: image is not JPEG")
    return audio, image


def resolve_case(manifest, case, cache_dir, archives=None):
    """Return the exact bytes to send for one case.

    The returned dict always carries `input_sha256`: a dict for multimodal
    cases, a single digest otherwise. `sent_bytes` are what the request body
    carries, which can differ from the downloaded bytes only by WAV rewriting.
    """
    kind = case["input"]["type"]
    archives = archives or {}
    if kind == "librispeech_utterance":
        archive = archives.get(kind) or ensure_librispeech(manifest, cache_dir)
        member = extract_members(archive, [case["input"]["member"]])
        wav = decode_audio_to_wav(
            member[case["input"]["member"]], case["input"]["wav_sha256"])
        if len(wav) != case["input"]["wav_bytes"]:
            raise DataError(f"{case['id']}: WAV byte count changed")
        return {"type": "audio", "audio": wav,
                "input_sha256": case["input"]["wav_sha256"],
                "source": case["input"]["member"]}
    if kind == "fleurs_utterance":
        archive = archives.get(kind) or ensure_fleurs_zh(manifest, cache_dir)
        member = case["input"]["member"]
        raw = extract_members(archive, [member])[member]
        if sha256_bytes(raw) != case["input"]["sha256"]:
            raise DataError(
                f"{case['id']}: archive member SHA-256 does not match the manifest")
        wav = float_wav_to_pcm16(raw, case["input"]["wav_sha256"])
        if len(wav) != case["input"]["wav_bytes"]:
            raise DataError(f"{case['id']}: WAV byte count changed")
        return {"type": "audio", "audio": wav,
                "input_sha256": case["input"]["wav_sha256"],
                "source": member}
    if kind == "dataset_row":
        question = _gsm8k_row(manifest, cache_dir, case)
        prompt = manifest["prompt_templates"]["gsm8k"].format(question=question)
        return {"type": "text", "prompt": prompt,
                "input_sha256": sha256_bytes(prompt.encode()),
                "source": f"gsm8k_test row {case['input']['row_index']}"}
    if kind == "image_url":
        image = _vqa_image(manifest, cache_dir, case)
        prompt = manifest["prompt_templates"]["vqa"].format(
            question=case["question"])
        return {"type": "image", "image": image, "prompt": prompt,
                "input_sha256": {"image": case["input"]["sha256"],
                                 "prompt": sha256_bytes(prompt.encode())},
                "source": case["input"]["url"]}
    if kind == "speechcoco_row":
        audio, image = _speechcoco(manifest, cache_dir, case)
        prompt = manifest["prompt_templates"]["joint"]
        return {"type": "audio_image", "audio": audio, "image": image,
                "prompt": prompt,
                "input_sha256": {
                    "audio": case["input"]["audio_sha256"],
                    "image": case["input"]["image_sha256"],
                    "sent_audio": sha256_bytes(audio),
                    "prompt": sha256_bytes(prompt.encode())},
                "source": {"audio": case["input"]["audio_source"],
                           "image": case["input"]["image_source"]}}
    raise DataError(f"{case['id']}: unsupported input type {kind}")


def resolve_cases(manifest, cases, cache_dir, log=None):
    """Resolve every selected case. Each archive is verified once."""
    archives = {}
    kinds = {case["input"]["type"] for case in cases}
    for kind, ensure, label in (
            ("librispeech_utterance", ensure_librispeech, "librispeech"),
            ("fleurs_utterance", ensure_fleurs_zh, "fleurs zh_cn")):
        if kind in kinds:
            archives[kind] = ensure(manifest, cache_dir)
            if log:
                log(f"{label} archive verified: {archives[kind]}")
    resolved = {}
    for case in cases:
        resolved[case["id"]] = resolve_case(manifest, case, cache_dir,
                                            archives=archives)
        if log:
            log(f"input verified: {case['id']}")
    return resolved
