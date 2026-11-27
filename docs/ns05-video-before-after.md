# NS-05 real long-video Before/After — Task #75 revision 1

## Decision

The bounded NS-05 golden-case evaluation completed, but the 16-frame second-resize
candidate has **no promotable default-path gain** and is **NOT_PROMOTED**. Commit
`44199d9fa323d3708074922ff63dd7ffc30ae69b` removes it from the final default
runtime. Candidate commits, raw evidence, rejected prototypes, and the revision-0
failure remain preserved.

Revision 0's frozen public-total result is unchanged: **2.3266% versus required
5%, FAIL**. The 9.73% serial temporal-16 input-stage result remains a historical
diagnostic, not a promoted end-to-end improvement.

- `NS05_GOLDEN_EXECUTION: COMPLETED`
- `CANDIDATE_PROMOTION: NOT_PROMOTED`
- `FINAL_DEFAULT_RUNTIME_CANDIDATE: REMOVED`
- `PHYSICAL_VOICE_PLAYBACK / PERCEPTUAL_QUALITY: NOT_RUN`
- `BARGE_IN / CROSS_MODAL_GENERATION: NOT_RUN`
- `GENERALIZED_QUALITY: NOT_COVERED`

## Traceable real multi-shot source

The selected media is a continuous 30–150 second interval of
`intel-iot-devkit/sample-videos` `driver-action-recognition.mp4`, upstream commit
`57978890822836f2b4743852f04f62fc511757e4`, licensed **CC-BY-4.0**. The full
418.4-second upstream and final derivative were visually reviewed before any model
response. The final 120.000-second H.264 derivative has 3,600 frames at 768×432 and
19,821,335 bytes, below the 32 MiB public limit. No looping, synthetic frames,
shot reordering, or model-generated substitution was used.

| Object | SHA-256 |
| --- | --- |
| Upstream source | `fd24e652c92a5759127aa9c7f23fea63d73aebb4bcea72763abe72ef06a20cf0` |
| Reviewed 30–150 s derivative | `ffef335edc161a2f6dead512e0fde96cf48fb997ae6b6491067e36c76eec58d0` |
| CC-BY-4.0 license | `e7edaaaeb0cd6da6a3dd97c54f3e8443ef027be685253fab64bf6eed2419ee7e` |

The [preregistration](../benchmarks/ns05-video/preregistration.json) fixes source,
questions, answers, scoring, schedule, and acceptance before inference. The
executable is `tests/ns05_video_live.py`; independent raw verification is
`tests/ns05_video_verify.py`.

## Default-input paired protocol

Official main `91310b36184c86a1c147debb27db8643cc0cb3e4` was merged as the second
parent of `d8a1bd3883793d5539b02f3a06ace1c335f9aeef`. Baseline execution is
`df563436b3255fec7aed955c4ff18fe06ef0c706`; candidate execution is
`c2b383561236151e347516a701093f2f194e51c3`.

Four fresh stacks ran baseline A → candidate B → candidate C → baseline D on
physical GPU 1. Each stack ran two cases with one warmup plus eight seeded,
interleaved measured repetitions: **72/72 requests**, 64 measured, and 36 exact
Before/After pairs. Every request omits `frame_selection`, uses historical default
8 frames, temperature 0, top-p 1, seed 510075, and logs `prefix_reused=0`.
Streaming voice uses the required `audio.format=wav` request envelope and produces
public `audio/pcm` chunks. There are no client policy/budget/scheduler controls.

Because the candidate only ran for non-exact 16-frame inputs, it could not execute
on this default-eight NS-05 workload. The measured result, not that structural
observation alone, decides non-promotion below.

Binary identities:

| Object | SHA-256 |
| --- | --- |
| Shared gateway | `ac9e81a42f1baa234497fddb9ec610ca8b0062c828c8a7459a16d9f5582cdab3` |
| Baseline VLM | `b82052b7ebdc96b17aecb795eb6034fbb84c2314612f91cc4f3bd11cc718900b` |
| Candidate VLM | `5dfea715cb282eba0cbe1fa4ba470893b062e8ea77b590e9ac6635009fb67ad2` |
| Shared ASR | `de572724fb2aa0a046f311724f671fa2a708d3e87c3a6d5f727ed81aa1259452` |
| Shared TTS | `fab9e01cfdf6114a485cbf7a0251f23790acb0444e3bbafdc4d3ccb354794539` |

## Decode/prefill and first public response

Each cell has 16 measured requests. Values are median / nearest-rank p95 /
inclusive IQR / MAD in milliseconds. No values were removed.

| Implementation | Case | Metric | Median | p95 | IQR | MAD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| baseline | phone-action | decode+prefill | 1241.863 | 1258.001 | 14.872 | 8.745 |
| baseline | phone-action | first text | 1864.292 | 1914.977 | 57.652 | 27.513 |
| baseline | phone-action | first voice | 2082.940 | 2134.801 | 56.664 | 26.374 |
| baseline | phone-action | public total | 2832.514 | 2885.049 | 57.573 | 25.466 |
| baseline | driver-scene-change | decode+prefill | 1238.976 | 1260.131 | 12.965 | 4.419 |
| baseline | driver-scene-change | first text | 1863.465 | 1910.227 | 57.492 | 27.512 |
| baseline | driver-scene-change | first voice | 2083.510 | 2130.230 | 57.682 | 28.863 |
| baseline | driver-scene-change | public total | 3483.461 | 3529.343 | 60.605 | 27.433 |
| candidate | phone-action | decode+prefill | 1244.662 | 1259.731 | 6.067 | 2.612 |
| candidate | phone-action | first text | 1852.126 | 1918.324 | 73.340 | 24.412 |
| candidate | phone-action | first voice | 2070.660 | 2138.154 | 72.944 | 24.477 |
| candidate | phone-action | public total | 2823.176 | 2889.075 | 71.104 | 24.615 |
| candidate | driver-scene-change | decode+prefill | 1245.236 | 1252.355 | 5.351 | 2.977 |
| candidate | driver-scene-change | first text | 1848.172 | 1908.105 | 64.747 | 17.564 |
| candidate | driver-scene-change | first voice | 2067.799 | 2128.135 | 64.687 | 17.372 |
| candidate | driver-scene-change | public total | 3469.417 | 3532.729 | 62.680 | 18.569 |

All first-text and first-voice p95 values pass the NS-05 <3 s threshold. The
largest p95 values are **1918.324 ms text** and **2138.154 ms voice**. One candidate
phone-action decode/prefill high outlier is retained; it is not removed.

Paired candidate-minus-baseline results:

| Case / metric | Median delta ms | Negative / positive | Sign-test p | Bootstrap 95% median delta ms |
| --- | ---: | ---: | ---: | --- |
| phone / decode+prefill | +3.347 | 8 / 8 | 1.0 | [-6.133, +9.055] |
| phone / first text | -8.363 | 10 / 6 | 0.4545 | [-14.792, +4.557] |
| phone / first voice | -7.360 | 10 / 6 | 0.4545 | [-15.499, +3.353] |
| scene / decode+prefill | +4.988 | 6 / 10 | 0.4545 | [-5.193, +9.895] |
| scene / first text | -4.998 | 10 / 6 | 0.4545 | [-11.263, +2.308] |
| scene / first voice | -6.302 | 9 / 7 | 0.8036 | [-13.523, +3.143] |

Both first-voice intervals cross zero, neither sign test is significant, and no
positive promotion gate is met. No TTFT/TTFA/decode-prefill median or p95
regression exceeds 1%; this non-regression does not turn no gain into promotion.

## Exact accuracy and retained misses

Prompts and scoring were not changed after answers. Each implementation/case has
16 measured answers; all answers, native semantics, and PCM bytes are exactly equal
within every baseline/candidate pair.

| Case | Baseline hits | Candidate hits | Retained raw outcome |
| --- | ---: | ---: | --- |
| phone-action | 0/16 | 0/16 | Every answer is `He holds it to his ear to talk.` It omits the frozen required `phone` token, so it is a scored miss even though the visual action is recognizable. |
| driver-scene-change | 0/16 | 0/16 | Every answer is `The driver stops talking on the phone and looks at the camera.` It misses the reviewed male-to-female driver change. |

These are real model quality misses, not transport failures and not evidence of a
runtime regression. No alias or prompt correction was made. Perceptual quality and
generalized accuracy remain unclaimed.

## Task #72 non-regression and independent evidence

The merged-main candidate separately reran the unchanged Task #72 gate:
320/320 requests, 304/304 parity checks, all **320 answers and native semantic
records exactly equal** to accepted baseline, and unchanged scores (default,
uniform, temporal-8 42/48; uniform/temporal-4 48/48). All 18 historical misses are
retained and this rerun has zero timing outliers. Independent verification fully
decoded all 165 source frames and checked 34 frozen natural plus 138 unchanged
Task #66 control files.

Task #72 candidate report SHA-256 is
`94e3973ccb01f53fa44f3c1709960d46fafba47db92c0133346b7c0eac500443`; its local
verification receipt is
`4e638e0707cd1cb0a5eab8e1bbfe5545b1377f65942f68df9e08336f97db7dfe`.

The external revision-1 bundle contains 1,115 files / 213,197,074 bytes, including
accepted runs, two excluded harness-preflight attempts, media/review images,
binaries, and natural evidence. The self-excluded manifest covers every raw file.

| Artifact | SHA-256 |
| --- | --- |
| NS-05 raw report | `85e516920dc87d188dcf02a347d6d79758b6737175f88acfbbcdbc6c3b749f23` |
| External `artifact-manifest.json` | `2162149bed6ba77386de1144caec23e631bf20cc274d22da8b5c6188443deeb8` |
| External `SHA256SUMS` | `40444a6edf40b571b3079a1e0274183b831fc98f3c4893e097f267d70b48ce2b` |
| Independent fresh-process verification | `e7214970d89559fc47adf7a58b443f348dc5d5592be1294ea38eec62ffd8459e` |
| Compact tracked scorecard | `231f38b439a15e88a9049444d18277d454d0a6823b04e8d8988ad80d8a3e8c88` |

The compact machine-readable record is
[ns05-video-before-after-sm90.json](evidence/ns05-video-before-after-sm90.json).
The revision-0 external bundle remains separate and unchanged. Final cleanup on
physical GPU 1 is 3 MiB with zero compute processes; only recorded task-owned
processes were stopped, and the resident service was never used or contacted.

Run verification with caller-supplied external paths:

```sh
python3 -I tests/ns05_video_verify.py "$NS05_EVIDENCE_ROOT" \
  --output "$NS05_VERIFY_OUTPUT"
```
