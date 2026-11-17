# Native stage integration — acceptance evidence

Run date: 2026-11-17 (container clock `Tue Nov 17 12:23 UTC 2026`).
Host: H20B, container `linyu-dev`, driver 535.161.08, CUDA 12.4,
cmake 3.31.1, g++ 13.3.0.

The service was started with `service/h20b-native-service.sh start`, driven by
`native/stage_tests/native_stage_smoke.py`, and stopped with the same script.
Every file here is text and at most 64 KiB. `smoke/smoke.log` is the harness
stdout of the run below.

## Deployment

| Piece | Process | GPU | Address |
| --- | --- | --- | --- |
| Server | `umm-infer serve` | — | `172.17.0.2:43400` |
| ASR stage | `umm-native-stage-asr --kind asr` | 2 | `ipc:///tmp/umminfer-native-ipc/asr` |
| Omni stage | `umm-native-stage-omni --kind omni` | 3 | `ipc:///tmp/umminfer-native-ipc/omni` |
| TTS stage | `umm-native-stage-tts --kind tts` | 5 | `ipc:///tmp/umminfer-native-ipc/tts` |

GPU0/1 keep the existing SGLang backends. The phase7 service on 43300 stayed
up throughout (`deploy/phase7_health.txt`). Nothing was published with
`docker port`.

Models, loaded from local directories (all report `UNKNOWN` as a git revision;
the directories carry no `.git`):

| Model | Directory | Size | `config.json` SHA-256 |
| --- | --- | --- | --- |
| Qwen3-ASR | `/root/omniinfer-model-validation-20261111-k3/models/asr` | 1.8G | `76d3ae46…5c21a478c` |
| Qwen3-Omni | `/root/omniinfer-model-validation-20261111-k3/models/omni` | 66G | `eab5093d…12d88f2e0d` |
| Qwen3-TTS | `/root/umm-mixed-output-20261116/models/qwen3-tts` | 2.4G | `81aca2b6…ab4729340c7455` |

The TTS checkpoint is byte-identical to the accepted TTS evidence
(`bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb`).

## Build and binaries

`deploy/build.txt`, `deploy/binaries.txt`, `deploy/proc_exe_sha.txt`. The
stages build as `native/stage_cli` with
`-DUMM_STAGE_CUDA=ON -DCMAKE_BUILD_TYPE=Release`; the server is the committed
`umm-infer` built with `zig build -Doptimize=ReleaseFast`.

| Binary | SHA-256 |
| --- | --- |
| `umm-infer` | `7a547c0c1e7da76c6adb243b887fdf46b386ccd2611b4d50c77a27ea813e3226` |
| `umm-native-stage-asr` | `f7634ed91fcf2d31a4387ee36943f9a0b2ce9bca46c25d976c5873c9e1dce79f` |
| `umm-native-stage-omni` | `23580f8be54601705436f17866aa4cc1109f0f89866d0cfe25f63346ec3f2d33` |
| `umm-native-stage-tts` | `c4daab07725482817c8e952da8027368b85c038d0747002e8ea8ac1ea3b3be31` |

`/proc/<pid>/exe` hashes of the four running processes matched these files.

`ldd` for each binary (`deploy/ldd_*.txt`) lists the CUDA runtime, libzmq,
libmsgpackc, libstdc++ and, for TTS, ICU. `deploy/python_torch_grep.txt`:

```
$ ldd .../umm-native-stage-* | grep -Ei 'python|torch|sglang' || echo 'no ...'
no python/torch/sglang libraries
exit=0
```

`deploy/processes.txt` is the process tree: one server and three stage
processes, no interpreter parent. Model load: ASR 4.8 s, TTS 6.2 s, Omni
17.9 s (19,218 tensors, 57.87 GiB resident).

## Live results (`smoke/`)

`smoke/smoke.log` is the harness stdout, `smoke/checks.json` the pass/fail
list (17 checks, all pass), `smoke/*.events.jsonl` the unified event traces
without base64 payloads.

| Check | Evidence | Result |
| --- | --- | --- |
| `/ready` 200, `in_flight` 0 | `smoke/health.json` | pass |
| ASR on the known WAV | `smoke/asr_direct.json` + `.meta/.stdout/.stderr` | pass: `Oh yeah, yeah. He wasn't even that big when I started listening to him. But in his solo music, didn't do overly well, but he did very well when he started writing for other people.` (46 tokens, 1.0 s) |
| text -> VLM | `smoke/vlm_text.json` | pass: `The sea is a vast, ever-changing expanse of saltwater that stretches to the horizon.` |
| image -> VLM | `smoke/vlm_image.json` | pass: `A man in a yellow shirt is standing on the back of a yellow SUV and using an ironing board to press a blue shirt.` |
| audio -> ASR -> VLM, SSE | `smoke/audio_to_text.json` | pass: one `asr_final`, then `complete` with a spoken answer |
| text -> TTS, SSE | `smoke/text_to_tts.json` | pass: one artifact, 24 kHz mono PCM16, 111,360 samples (4.64 s), rms 0.0711, non-silence 0.621 |
| TTS -> native ASR round-trip | `smoke/tts_roundtrip.json` | pass: `Hello, welcome to the native inference engine test.` |
| cancel ASR / Omni / TTS | `smoke/cancel_{asr,omni,tts}.json` | pass: `/abort` 200, `cancelled` terminal, no artifact, `in_flight` 0 |
| request after cancel | `smoke/after_cancel.json` | pass: `OK.` |

`smoke/asr_direct.meta` holds the exact `curl` command, `process_exit: 0` and
`http_status: 200`; `.stdout` is the response body and `.stderr` is empty.

## Cancellation

`POST /abort` broadcasts the existing `AbortMessage`; the wire protocol is
unchanged. The stage scheduler binds the flag to the request id and each engine
raises `RequestAborted` at its own boundary. The stage writes one line per
abort (`deploy/stage_logs.txt`):

| Engine | Boundary reached | compute_ms at abort | abort -> compute exit |
| --- | --- | --- | --- |
| ASR | `aborted in prefill layer 0` | 194 | 0 ms |
| Omni | `aborted at token 0` (vision and prefill done) | 15,425 | 0 ms |
| TTS | `aborted in codec decoder block 1` | 12,677 | 0 ms |

Each of the three aborts produced no result (`kinds` has no `artifact` or
`complete`), `in_flight` returned to 0, and the next request completed. The TTS
abort landed in the waveform decoder, which runs about 30 s of the 40 s
synthesis; the decoder now checks between the pre-conv, every pre-transformer
layer, every upsample stage, every decoder block and every residual unit, and
drains the stream at each check so the boundary reflects real progress.

## Notes and limits

- Request `params` do not reach a C++ `Stage` compute function. Each stage uses
  its configured cap; the H20B launcher sets Omni to 96 tokens because the
  accepted engine decodes about 2 s per token.
- The round-trip needs a 24 kHz -> 16 kHz conversion because the ASR stage is
  16 kHz only. The harness uses a Hann-windowed sinc; a plain 3-sample average
  made the ASR misread a correct utterance, and the scipy polyphase resampler
  from the accepted TTS evidence gives the same transcript on the same file.
- Inside one codec stage there is no finer boundary: the longest single step is
  one decoder block, about 3 s.
- `host_tests.txt` is the macOS host build and unit test of the wire adapters
  and of the scheduler cancel semantics.

## GPU state

While loaded (`deploy/gpu_while_loaded.txt`): GPU2 2387 MiB, GPU3 78427 MiB,
GPU5 5201 MiB, GPU0/1 unchanged. After `stop`
(`deploy/gpu_after_stop.txt`): GPU2/3/5 back to 0 MiB; GPU0/1 keep their
existing backends. `deploy/processes_after_stop.txt` lists no live stage.

## Four-stage update: native Image on 43301

The later `four/` run adds `umm-native-stage-image` on GPU4 and moves the
candidate server to `172.17.0.2:43301`. The original three-stage evidence above
is kept unchanged. The four-stage launcher starts ASR, Omni, Image and TTS; the
candidate was stopped after acceptance.

`four/checks.json` records 22 passing checks. `four/mixed_all.json` is one
request with text, image and audio inputs and with text, image and audio
outputs. Its native PNG is 512x512 with 15,130 colors and 501,668 bytes. Its
native audio is 24 kHz mono PCM16, 374,400 samples (15.6 s), RMS 0.0416.

| Binary | SHA-256 |
| --- | --- |
| `umm-infer` | `7a547c0c1e7da76c6adb243b887fdf46b386ccd2611b4d50c77a27ea813e3226` |
| `umm-native-stage-asr` | `52e760c5a94675ce95d5e6fa8caef42a3d56d6e226f537e37bbaece1d57ff9ba` |
| `umm-native-stage-image` | `92b1836eb6f2287c3297449383558bf44ad3273a48166ff99cac0f652c6d619e` |
| `umm-native-stage-omni` | `a1a693bcc478054c8b14426095b460cf56e896d02bdb3cea19449f61ebb9171c` |
| `umm-native-stage-tts` | `0b7fb87df20197391be24bcce60352495edd7221ed97083c89b81920adf12211` |

`four/deploy/ldd.txt` lists no Python, PyTorch or SGLang runtime dependency.
The public wire shape is unchanged: the Image stage returns a map with
`media_type=image`, `mime_type=image/png`, `format=png`, size, seed, steps and a
`data` Bin. The accepted Image slice currently supports 512x512 PNG only.

| Cancellation | Boundary | compute_ms at abort | abort -> compute exit |
| --- | --- | --- | --- |
| ASR | `aborted in prefill layer 0` | 193 | 0 ms |
| Omni | `aborted at token 26` | 512 | 0 ms |
| Image | `denoise step` | 18,262 | 1,107 ms |
| TTS | `aborted in codec decoder block 1` | 12,637 | 0 ms |

All four cancellations settled as `cancelled`, produced no artifact, returned
`in_flight` to 0, and a later request returned `OK.` See
`four/deploy/abort_boundaries.txt` and `four/cancel_*.json`. The Image boundary
synchronizes once per denoise step, so an abort cannot leave a fully queued
trajectory running behind it.

After the candidate stop, `four/deploy/gpu_after_stop.txt` records GPU2/3/4/5
at 0 MiB. GPU0/1 were not changed by the candidate. PNG and WAV artifacts are
kept on H20B, not committed; their parsed statistics and event descriptors are
in `four/`.
