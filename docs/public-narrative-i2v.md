# Public VLM narrative image slot → explicit native I2V gate (Task #65, 2026-11-27)

## Result and scope

**CONTROLLED_LIVE / PASS** on one explicitly idle NVIDIA H20 SM90, physical GPU
1, using official Wave 88 main
`2b0403b455d713af2b1e3246106368ab3d6d8989`. That official base contains the
accepted Task #62 native anchor validation and direct FLUX.2→Wan2.2 continuity
records; no earlier Task #62 ancestry claim is needed or made.

One public `POST /v1/chat/completions` text request produced one standalone
FLUX.2 image slot, then an explicit `[[video: … | source=slot-0]]` directive and
one native Wan2.2 I2V MP4. Streaming native TTS spoke the clean prose. This was
public HTTP/SSE → `UnifiedEngine` → real VLM/output-stage orchestration, not a
direct-stage substitute. The required legacy entry was the existing CPU echo
stage solely for readiness; the text-only plan never submitted work to it.

The serial Qwen3.8 VLM stayed resident while FLUX.2, native TTS, and native Wan
ran **one output stage at a time**, in that order. Every task-owned process
returned 0, final `/health` was `in_flight=0`, and GPU 1 returned to `3 MiB`.
No TTS implementation, online optimizer, Qwen video-selection code, resident
service, public schema, scheduling threshold, or RGB threshold was changed.

The only runtime change is a cancellation-lifecycle fix evidenced by a failing
raw deterministic record: `UnifiedEngine::finish_locked` now marks both pending
and running narrative slots `aborted` on cancellation. All other changes are
focused tests, evidence tooling, and documentation.

## Exact execution identities

| Object | Identity |
| --- | --- |
| Official base | `2b0403b455d713af2b1e3246106368ab3d6d8989` |
| Runtime cancellation fix | `4573e6bcc5bf100a708cbca024355a08bf1296cf` |
| Server/native binary build tree | `cc654200f6fdbc1bf89759c8518ac4d7f5f827b6` |
| Public-run harness execution | `452706a72faecc4bbf7b9e6344f7c13ad1ad0e0f` |
| Final independent verifier | `bbf23322d475e47c4d9054c2b9fe9c6b8d8a4f85` |

The binary build tree already contained the runtime fix. Later execution
commits changed only the live/verification harness, not server or native-stage
sources. Exact binary and source hashes below bind the executed artifacts. The
producer verifier's original hash is retained separately from the corrected
independent verifier's hash; neither is aliased to the other.

**Portable launch-default correction.** Review found a task-specific loader
path embedded as a tracked default. Commit
`0e4eab8ca275d24cbcea135c398e945ac24dc671` changes only `--library-path` to an
empty portable default that inherits the process environment/system loader;
callers may still provide an explicit path externally. The final portable live
source SHA-256 is
`b3593e1c13c0140980ef3e68574f31921b6eafa0e8bd1b1c1c3a3227a7a025d3`.
The executed-harness SHA-256 remains the historical identity
`3cc1de61e8470ca307a31c09b9cab649efdb381b6443fd386337b251ff5e65ef`.
Sealed `report.json`, raw SSE/CAS, media, and the exact five-leg PCM/WAV join
are unchanged. No new GPU generation was used for this launch-only correction;
fresh raw-evidence verification was rerun with the final independent verifier;
its separate verification JSON SHA-256 is
`d429166a8886ef5d0677bf3b8568bfce44cb96359c449cd0434096511a498669`.

## Public request, stream, and clean output

Frozen settings were seed 42, temperature 0, image 512x512/4 steps, video
704x480/29 frames/16 fps/20 steps/guidance 5.0, and native streaming PCM16 TTS
at 24 kHz mono. The public request SHA-256 was
`ec2e3167359ccd916adeba101311154e587433d30f48425b55adf9558fd8c8b6`.

The stream contained **69 unified events**, **71 raw frames**, **111 transport
chunks**, and exactly one final `complete` plus one `[DONE]`. Unified sequence
numbers were gap-free. Two transport-only OpenAI chunks had no `unified`
envelope and were retained and hash-checked without being counted as unified
events. Event counts were:

| Event | Count |
| --- | ---: |
| `vlm_delta` | 34 |
| `visual_slot` | 2 |
| `artifact` | 32 |
| `complete` | 1 |

There were **32 complete artifacts in 32 chunks**: one PNG, one MP4, and 30 PCM
chunks. Raw SSE was 1,617,694 bytes, SHA-256
`8a62a2807c6393fb4523cc1401aac0137293d8a778eb0cc953a0d779453267c4`.
The schema-v2 trajectory JSONL SHA-256 was
`f2e6d51c60dbeb4961d7d2b148da773b287e7b910655be7f53f262716e8b668d`.

Clean prose, SHA-256
`84154ab5f11c50a5e2346d86bb161b5fe4ae20a45cd0100f53fbfe43127e05c8`, was:

> A single red lantern floats above the misty blue mountain lake at dawn.
>
> The lantern sways gently as the camera slowly orbits above the misty lake.

Neither the image/video directives nor `source=slot-0` appeared in clean prose,
render prompts, or exact TTS text inputs.

### Visual-slot and artifact identities

`slot-0` was an image at clean-text offset 72. `slot-1` was a video at offset
148 with `continuity_mode=i2v_chain`, `anchor_slot_index=0`, and
`anchor_explicit=true`; the time-dependent digest was correctly absent from the
`visual_slot`. The final video artifact and terminal leg carried the verified
source digest. TTS, image, and video terminal legs were all `completed`.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| FLUX.2 slot-0 PNG | 301,844 | `01af1c29f95aa2362b57067bd6592a032617b8c46d91509e7b946d7b0d4e150b` |
| Native Wan slot-1 H.264 MP4 | 140,397 | `49c6ae4e5460a17a387fb5e31f821a0739cf5e789a21419fc125a63194110b60` |
| Public PCM, 30 chunks total | 748,800 | Per-leg digests below |

The video is H.264/yuv420p, 704x480, 29 frames at 16 fps.

## Immutable source binding and native dispatch

A read-only ZMQ tap forwarded the native submit unchanged and retained its raw
MessagePack. The packed submit was 302,597 bytes, SHA-256
`55341d07481825de1e7d22cf4eabceca48a186a52c555a9e7ad97112905ba315`.
Independent MessagePack decoding proved that its binary `image` was exactly the
301,844 public source bytes, and that
`anchor_sha256=01af1c29…150b`, `anchor_slot_index=0`,
`anchor_explicit=true`, and `anchor_crop=center_cover`. No source suffix reached
the native render prompt.

The tap's observed/forwarded timestamps describe **queueing**, not native
receipt. The queued submit preceded client observation of the final source SSE
by 137 ms; it did not execute early. Wan process start and readiness followed
public source completion, and the native log contains exactly one
`backend=native-cuda-wan`, `i2v=1` request for `http-1:slot:1`, with no T2V
execution. Thus the retained evidence proves source settlement before native
I2V dispatch without mislabeling an outbound queue timestamp as device receipt.

## Exact per-leg speech join

Five TTS submits produced 30 public PCM chunks. A second read-only tap retained
every native stream message and every terminal WAV. For each leg, native chunk
indices were joined to public `stream_index` values in order. Every public PCM
chunk matched the corresponding native bytes, the concatenation matched the
terminal WAV PCM exactly, and no missing or extra public chunk remained. All
five native logs reported `identical=1`. The Coordinator barrier before Wan was
`submitted=5, completed=5, failed=0, aborted=0, in_flight=0`.

| Leg | Public chunks | PCM bytes | PCM SHA-256 | Terminal WAV SHA-256 |
| --- | ---: | ---: | --- | --- |
| `tts:0` | 2 | 42,240 | `fcfc9b27102559d23cbf92c3b074738102b49620f0b6d50d164952da7a10f8c2` | `af58308a3b1f18e46ea222eef7b4b100b09565c9f7badc55125a0c8539e9a656` |
| `tts:1` | 7 | 188,160 | `a34a884c2a3c9a7d048543f8b633ba3e554ca9a89c6285f2913f23e693cd64a1` | `d82fd93f43b181ea3b474c915d3a2d29a0702044f1587a249d0642f56c3679c0` |
| `tts:2` | 4 | 84,480 | `3bb7be676913fa0564bc0db7d0b3edc5dffdc6ac8da67fc5269c5a05465b2674` | `7fa90465817118eb3178635775cf5470c19d19ec1940aecf2748ea745eb0e921` |
| `tts:3` | 10 | 264,960 | `5b4a940f05de7382a4fd67ebc87dd33b1cbb9e56127bf2d471b63efdfcc8d4b6` | `4751dd37d22a8c97ddecc25eafe5e24e046e6e60df93ea561cd25da9e7947891` |
| `tts:4` | 7 | 168,960 | `f1e612921eb397f40e791ed80c45e183a66e58e0fba30406d60a53c8eb199787` | `16263c6b91517aeab48b24666b4f4fddff5c5dd9dd3013b0c0b29c95d2b5f015` |

The ordered TTS text-token sequence equaled clean prose. Only intentional
leading/trailing whitespace at sentence/clause span boundaries was normalized;
no directive or source suffix was normalized away. All PCM chunks were
non-silent, 24 kHz mono PCM16. Individual chunk hashes and RMS values are in the
machine-readable evidence.

## Frozen RGB continuity/motion proxies

Metric version `center-cover-64x40-rgb8-frame0-frame28-v1` and thresholds were
unchanged: first-frame MAE <= 48.0 and motion MAE >= 1.0. FFmpeg performed
aspect-preserving scale plus center crop for evaluation RGB; full-resolution
RGB was retained independently.

| Measurement | Value |
| --- | ---: |
| Source/evaluation first-frame MAE | **2.259115** |
| Frame 0/frame 28 motion MAE | **5.124479** |

| RGB buffer | Bytes | SHA-256 |
| --- | ---: | --- |
| Source full RGB 512x512 | 786,432 | `86d37ef826cc41380b59a6d49e2d6e52537dc4395f8d1ac725d2d58fde50e9a5` |
| Source evaluation RGB 64x40 | 7,680 | `a0af3c6042c3cc67cf38bf72298bc76fe1ff68aa329f6afcc0104705e164dd39` |
| Video first full RGB 704x480 | 1,013,760 | `fd5a6349034a7e6ee8c0ce55bddae3a22a6dd7a9e3f36e6feef3eb61ab3ad5b0` |
| Video last full RGB 704x480 | 1,013,760 | `ae7e1675f15beeb5dae545e5b587e1ad299759b17439c02141a4c1c81df7af3e` |
| Video first evaluation RGB 64x40 | 7,680 | `2c268edbe8384a43f74469984dd412081ad1b7d89356b64066eeb247bf3e71bf` |
| Video last evaluation RGB 64x40 | 7,680 | `061795604ebb355c316e953ccdcbcb8a053279425df7a9ff9db3cae478b2713b` |

These are byte/continuity/motion proxies only. **PERCEPTUAL_QUALITY and human
visual review remain NOT_RUN.**

## Model, config, binary, and runtime provenance

| Object | Files | Tree or object SHA-256 |
| --- | ---: | --- |
| Qwen3.8-27B model tree | 29 | `22e638404d0e413007064c47d5c98ad1696a26a0e980eb2fce83cbbfcca0a304` |
| FLUX.2-klein-4B model tree | 25 | `10c1b2d7096cd6c2798fa9d2b9aecfab5163117cbf3d66de6f32f9a2e8f17997` |
| Qwen3-TTS-1.7B-CustomVoice tree | 14 | `69c90c52d9d88390f4b4f540b09acf40f5ead7080650585ff20d2e144aa5e69d` |
| Wan2.2-TI2V-5B-Diffusers tree | 30 | `b118ac07b533cf7d68a4806ec70bdf39041515673e44535afeda5cfefe761616` |
| `umm-infer` binary, 54,825,024 B | — | `5a76d96d46d97f648d00eeb1007acbf100b429b9b5506fb006cdf43281dbd8a9` |
| Qwen3.8 stage binary, 3,630,536 B | — | `7ca5a36528f34a71e6c5092aae9e5fe4cb5211e4b1daa3530fa31b19be55fc02` |
| TTS stage binary, 3,217,424 B | — | `117f93722c9a584e98301bf0c5d38a389301a64fccd2544bcd5b7d1a5602a4e4` |
| Wan stage binary, 3,171,472 B | — | `000a78196e5d14f369aa82c5f1b5886dacfc4be532cab570791179b59ccda6bf` |
| FLUX Stage source | — | `4efe0f79ec227540382ddb64486a28dd1675159d2c5263964bdc9091cfd307d0` |
| Live harness at execution | — | `3cc1de61e8470ca307a31c09b9cab649efdb381b6443fd386337b251ff5e65ef` |
| Final independent verifier | — | `8296665afc393d2ebcd29d8416808080ca92233e48e65716c133c7b289e15e5a` |

Config identities included Qwen3.8 `config.json`
`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`,
FLUX `model_index.json`
`51a76cb1cf3ed37423a1128c79c22faee8e6fbe7f5aaeb737f0a258930dbaac0`,
TTS `config.json`
`17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9`, and
Wan `model_index.json`
`6a72faeb564b0e894aea8fc4ef27241106eb739e8584e869b61589a65473add7`.
Every individual model file hash is in the evidence JSON. Model revisions are
`UNKNOWN`; local file/tree hashes are the exact identity, not an upstream
revision claim. The runtime used Torch 2.6.0+cu124, Diffusers 0.43.0,
Transformers 5.6.0, pyzmq 27.3.0, and msgpack 1.2.4.

GPU 1 memory was 3 MiB before, 55,613 MiB at VLM readiness, 71,226 MiB with
FLUX, 56,365 MiB after FLUX stopped, 62,196 MiB with TTS, 56,365 MiB after TTS
stopped, 80,516 MiB with Wan, 84,754 MiB after completion before cleanup, and
**3 MiB after cleanup**. These are lifecycle observations, not a performance
benchmark.

## Negative controls and the evidenced cancellation fix

Deterministic orchestration, native parser, and raw-replay controls covered:

- malformed, missing, duplicate, forward, self, wrong-modality, wrong-target,
  and cross-session explicit references: fail closed with no generation or
  stale bytes;
- wrong source/anchor hashes and source tampering: replay and native parser
  rejection with no generation;
- failed explicit sources: dependent slot fails without implicit T2V fallback;
- cancellation while the source is pending: one `cancelled` terminal, no image
  or video artifact, no late submit, and no events after the terminal;
- implicit `off`, `prompt_anchor`, latest-image chaining, no-image T2V, and
  failed-anchor T2V fallback: unchanged under the full native suite.

The added cancellation regression first retained
`slot-0=running, slot-1=aborted` after the cancellation terminal. A primary
abort could finish before a running source observed its abort, after which the
Coordinator correctly dropped late completions and left the public state stuck.
Commit `4573e6b` marks pending **and running** slots aborted. The same raw case
then retained `slot-0=aborted, slot-1=aborted`; late successful source bytes and
post-terminal VLM/directive events remained suppressed. No public field or
scheduling rule changed.

## Independent replay, failed attempts, and evidence checksums

The corrected fresh verifier did not import the live driver or replay producer.
It independently parsed raw SSE and MessagePack, reconstructed every artifact,
rechecked source/anchor and per-leg PCM/WAV binding, decoded RGB, inspected
native logs and monotonic process timing, and rehashed all binaries and model
files. It passed 69 events, 32 artifacts, five TTS legs, and 30 PCM chunks. A
second fresh local replay then produced byte-identical JSONL and CAS and passed
the narrative source-binding validator.

Raw SSE, CAS, native MessagePack, media, RGB, logs, and the independent
reconstruction remain outside the tracked tree at
`/tmp/umminfer-public-narrative-65-final-evidence/`.

| Evidence object | SHA-256 |
| --- | --- |
| `report.json` | `2b7272bc4de5d29b3289333b0421ae5ff484d8eebc17eb61764f8cd71402cb7f` |
| `independent-verification.json` | `6240984694f67adb38c0d6681d7467feb50cd9a95157f68ce51f88c165ff58c4` |
| `summary.json` | `f9f44df2f5242e29247973532ff3527cbe44b1c76a81b51b57e95102d70e24f5` |
| `SHA256SUMS` (141 files) | `3bfb3e09fccbbba639f3bfb506d063a57f2e3c3aa9464742daaf3c159fb29510` |

Failed attempts were retained separately and are not counted as passes:
`cc65420` failed before public generation because the legacy entry was absent;
`0426a04` stopped TTS before the second native completion and was rejected by
the exact join; `5efd13e` completed its public stream but failed the older text
assertion. Its retained five-leg stream subsequently passed the final exact
PCM/WAV and token checks. The accepted `452706a` stream passed live, then the
independent verifier was corrected for plain transport chunks and the
queue-versus-native distinction and rerun without regenerating media.

## Reproduction and tests

With exact local assets, an empty output directory, and GPU 1 idle. The default
loader is portable and inherits the environment; when native dependencies are
outside the system loader, supply `--library-path /path/to/deps/lib:/path/to/curl/lib`
explicitly in the launch environment:

```sh
UMMINFER_LIVE_OPT_IN=1 /path/to/venv/bin/python -I \
  tests/public_narrative_i2v_live.py \
  --python /path/to/venv/bin/python \
  --server-binary /path/to/umm-infer \
  --vlm-binary /path/to/umm-native-stage-qwen38 \
  --tts-binary /path/to/umm-native-stage-tts \
  --wan-binary /path/to/umm-native-stage-video \
  --flux-stage python/umminfer_flux2_stage.py \
  --vlm-model /path/to/qwen3.8-27b \
  --flux-model /path/to/FLUX.2-klein-4B \
  --tts-model /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --wan-model /path/to/Wan2.2-TI2V-5B-Diffusers \
  --out-dir /tmp/umm-public-narrative-65-reproduction \
  --base-commit 2b0403b455d713af2b1e3246106368ab3d6d8989 \
  --execution-commit 452706a72faecc4bbf7b9e6344f7c13ad1ad0e0f
```

Offline checks and final results:

```sh
python3 -I tests/public_narrative_i2v_live.py --self-check
python3 -I tests/narrative_slot_reference_live.py --self-check
python3 -I tests/flux2_i2v_continuity_live.py --self-check
python3 -I tests/sse_replay_test.py
python3 -I tests/flux2_stage_contract_test.py
umm-test-local
```

- Full `umm-test-local`: **PASS**.
- Native stage host CTest: **9/9 PASS** with Makefiles; the initial Ninja
  configuration was unavailable and ran no tests.
- Remote `stage_video_common_test`: **PASS**.
- Raw replay tests: **10/10 PASS**.
- FLUX contract tests locally: 14 tests, 12 pass and two dependency skips; the
  remote exact-PCM self-check passed with tamper/extra-chunk rejection. The
  local exact-PCM self-check was `NOT_RUN` because msgpack is unavailable.
- Feature-map anchor validation is reported in the final integration commit.

## Explicit limits

This is one real public narrative request, not a generalized quality or
performance study. Perceptual quality, human visual review, generalized model
quality, production deployment, and cross-host/generalized performance remain
`NOT_RUN` or `NOT_COVERED` as applicable. No resident-service validation or
deployment claim is made.
