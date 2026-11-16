# Native Qwen3-TTS 0.6B — acceptance evidence manifest

Host: H20B, container `linyu-dev`, dedicated GPU5 (`CUDA_VISIBLE_DEVICES=5`).
No Python/PyTorch/SGLang/qwen_tts on the TTS runtime path; Python used only
for the test-only resampler and offline reference generation.

## Build / environment

- CUDA toolkit 12.4 (V12.4.131), NVIDIA driver 535.161.08, target sm_90.
- Binary `umm-tts-native` SHA-256:
  `8c91f5a7c1c8e154155bf0c282757a90659e9dbce178b1dfecdad75f14e9255c`.
- `ldd` dynamic deps: libcudart.so.12, libicuuc/libicui18n/libicudata,
  libstdc++/libc/libm/ld, libdl/librt/libpthread. No PyTorch/SGLang/qwen_tts.
- Model revision (HF download metadata commit):
  `85e237c12c027371202489a0ec509ded67b5e4b5`. Per-file SHA-256 in
  `model_files.sha256`.
- GPU5 peak resident memory: 4877 MiB (`gpu5_peak.txt`). GPU5 reports 0 MiB
  after every run (`gpu5_after_release.txt`, `gpu5_after_asr.txt`,
  `{en,zh}.gpu_before/after`).

## End-to-end TTS (final WAV, this directory's hashes are authoritative)

| Lang | Text | Frames | Duration | RMS | Peak | Non-silence | Wall s | Exit |
|---|---|---|---|---|---|---|---|---|
| en | Hello, welcome to the native inference engine test. | 88 | 7.04 | 0.0422319 | 0.385009 | 0.764234 | 46.794 | 0 |
| zh | 其实我真的有发现，我是一个特别善于观察别人情绪的人。 | 83 | 6.64 | 0.0551875 | 0.452735 | 0.74498 | 43.988 | 0 |

Both are real 24 kHz mono PCM16 WAV. Raw CLI stdout in `{en,zh}.tts.log`,
wall time in `{en,zh}.tts.timing`, exit code in `{en,zh}.tts.exitcode`.

WAV SHA-256 (`wav.sha256`):

- en.wav `c0529f8752d2d97883d7494b26e4db2f1e400d73636e7376bb4f3537d6b3c073`
- zh.wav `79271212bb2dc7f1e388ec2851573c9101da9644563bebec77943151a448adb2`

## ASR round-trip (native umm-asr-native, GPU5)

24k->16k test-only resample via `native/tts_tests/tools/resample16.py`
(scipy polyphase), command in `{en,zh}.resample.log`. ASR command, input
text and exit code in `{en,zh}.asr.meta`; transcript in
`{en,zh}.asr.stdout`; engine stats in `{en,zh}.asr.stderr`; TTS WAV and 16k
input hashes in `{en,zh}.asr_inputs.sha256`.

- en: exit 0, transcript "Hello, welcome to the native inference engine test."
- zh: exit 0, transcript "其实我真的有发现，我是一个特别善于观察别人情绪的人。"

Both match the input text verbatim.

## Sanitizers (compute-sanitizer, GPU5)

- memcheck: `sanitizer_memcheck.log` — ERROR SUMMARY 0 errors, exit 0.
- initcheck (codec path): `sanitizer_initcheck_codec.log` — ERROR SUMMARY 0.
- racecheck: `sanitizer_racecheck.log` — RACECHECK SUMMARY 0 hazards
  (0 errors, 0 warnings), exit 0.

## Host tests (macOS, no CUDA)

`tts_host_tests <tokenizer-data-dir>` — `host_tests.log`, all passed, exit 0.
BPE fuzz parity 30010/30010 was recorded during bring-up.

## Log size

Every persisted log is hard-capped at 65536 bytes with `head -c 65536`.

## Notes

- Talker codec token traces are bit-identical run to run (deterministic
  content). The FP32 codec reduces in nondeterministic block order, producing
  sub-LSB PCM16 waveform differences across runs; stats and ASR are stable.
  See `native/tts/NOTICE` and the TTS_NATIVE.md limitations section.
- Binary WAVs stay on the H20B host under the final/ directory; this
  repository commits only the text evidence above (WAV git-ignored).
