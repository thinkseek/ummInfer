# NS-02 exact-CUDA Wan2.2 interactive-video speedup (Task #89, 2026-11-28)

## Result and promotion boundary

**CONTROLLED_LIVE / NOT_PROMOTED.** On one explicitly idle NVIDIA H20 SM90,
physical GPU 0, the frozen fixed-algorithm cuBLASLt Wan DiT candidate reduced the
median paired public complete-artifact latency by only **65.863169 ms
(0.593504%)**. Three of 15 pairs regressed, every measured After artifact was at
least 10,969.267294 ms, and all 15 pairs failed exact MP4 and exact decoded
first-frame RGB parity. The candidate therefore failed the speed, per-artifact
5-second, and exact-fidelity gates. It is absent from the final default runtime.

Candidate commit `b64f022a3555c925982a9d88274d98c82edddd9f`, execution/protocol
commit `9293fb1aaf8e445ddb493f0d21c6347447bdb293`, and candidate-removal commit
`5c74ee1bb6c8a48a1626742de6f009290661f493` retain the complete lineage. Final
`native/` and `src/` are byte-identical to official main
`e084d8404b1d96bee5cd6f83d8299c2498a3af65`; that official runtime is also
source-identical to the Task #86 raw-control runtime in those directories. No
successive candidate was tried after observing the frozen outputs.

The candidate did not use TeaCache or any other approximate cache, alter solver
arithmetic, reduce precision, change sampling or codec settings, or modify public
schemas. Human/perceptual approval remains **NOT_RUN**; SSIM, PSNR, RGB, and exact
byte controls are not a subjective quality review. No deployment, training,
optimizer, or resident-service validation is claimed.

The compact independently verified scorecard is
[ns02-exact-cuda-speedup-sm90.json](evidence/ns02-exact-cuda-speedup-sm90.json).
Raw media, RGB, MessagePack frames, process logs, manifests, and verification
outputs remain external only.

## Candidate selection and profiling

The accepted Task #86 profile attributed about 8.8 s to DiT, 2.0 s to VAE decode,
and 0.12 s to MP4 encode. New CUDA graph-node profiling on GPU 0 measured two
baseline end-to-end runs at 8,866.592 and 8,899.836 ms DiT and produced the exact
frozen wide MP4 hash. The graph profile assigned approximately 6.7 s to the large
DiT projection GEMM families and 1.593864 s to 1,800 self/cross FlashAttention
kernel instances. Host/device transfer and public handoff were too small to offer
a multi-second exact optimization on their own.

One candidate was selected: replace only Wan DiT BF16 GEMM dispatch with cached
cuBLASLt plans using frozen heuristic indices 2 by default, 5 for the
K=14336/N=3072 FF2 shape, and 0 when the preferred index is unavailable. BF16
inputs/outputs, FP32 compute, matrix layouts, all other CUDA arithmetic, UniPC,
TeaCache-disabled behavior, CUDA Graph semantics, and lifecycle behavior were
unchanged. The frozen protocol is
[preregistration.json](../benchmarks/ns02-exact-cuda-speedup/preregistration.json).

A deterministic synthetic-shape probe found zero output-bit mismatches between
legacy cuBLAS and the selected plans, but projected less than 0.1 s aggregate
improvement. A real-weight direct A/A control then produced candidate MP4
`28ac2f4ee15aa41d621930ac5e97cd0b4683dc20722ae4e7754da9e3b4e01d8f`, not the
baseline `6e487e7fe7fca4c439658ccb861431c21f56d7e7619768869517ef115f33e934`;
DiT moved from 8,866.592 to 8,794.343 ms. This was negative preflight evidence,
not an accepted exact-parity result. The candidate was not tuned afterward. The
frozen 15-pair gate was retained to measure the complete public latency and
lifecycle outcome and to verify, rather than assume, exact parity.

## Frozen NS-02 protocol

The protocol reuses the accepted Task #86 raw controls and binds baseline profile
report `73600dc655ecdd938ba61a20177c5db3cd58d3b557eaaa74fa22f0657b803e1e`.
It fixes the exact real FLUX.2 sources, seed 42, 29 frames, 20 UniPC steps,
16 fps, guidance 5.0, five process-separated warm pairs per geometry, one excluded
warmup per shape in each fresh arm block, and seeded arm order `79025`. Both arms
omit `teacache_thresh` and execute with effective TeaCache 0.0 and 0/40 hits.
Every observation and outlier remains in all statistics.

The exact-CUDA v3 protocol adds two requirements without weakening the old gates:
paired baseline/candidate MP4 bytes must match, and every baseline must match the
frozen-profile MP4. The latter passed 15/15; the former failed 15/15. Exact RGB,
SSIM >=0.90, all-frame PSNR >=24 dB, temporal-motion ratio [0.5,1.5], source
continuity, nonzero motion, decoded H.264/yuv420p geometry, 29 frames, 16 fps,
cancellation, malformed-input, recovery, and terminal invariants all remain
active. The primary metric is median paired candidate-minus-baseline public
complete-artifact latency, with at least a 2,000 ms improvement and all paired
and per-shape deltas negative required for promotion.

## NS-02 paired public latency

The table uses complete public artifacts. Public playable and complete-body
timings are both retained in the raw report. All 15 After observations exceeded
5 seconds. The median paired change was **−65.863169 ms (−0.593504%)**, with 12
negative and three positive deltas; this is not the preregistered repeatable
multi-second speedup.

| Geometry | Before median / p95 / max (ms) | After median / p95 / max (ms) | Median paired delta (ms) | Every After <5 s |
| --- | ---: | ---: | ---: | --- |
| Wide 704×480 | 11,107.427034 / 11,218.359571 / 11,224.733002 | 10,995.362513 / 11,166.118531 / 11,174.084020 | −58.609269 | **FAIL** |
| Tall 480×704 | 11,089.464638 / 11,179.714429 / 11,200.031760 | 11,012.620608 / 11,151.503556 / 11,155.837297 | −65.863169 | **FAIL** |
| Square 512×512 → 704×480 center-cover | 11,097.341163 / 11,194.412437 / 11,194.607374 | 11,101.680520 / 11,203.543202 / 11,212.947414 | −78.960879 | **FAIL** |
| **Pooled** | **11,097.341163 / 11,207.442133 / 11,224.733002** | **11,012.620608 / 11,185.743038 / 11,212.947414** | **−65.863169** | **FAIL** |

The positive paired observations were wide +66.656986 ms, tall +57.392191 ms,
and square +120.577239 ms. Retained paired-delta IQR outliers were wide
+66.656986 and −255.465708 ms plus square +120.577239 ms. No After IQR outlier
occurred. Retained Before tall outliers were 11,200.031760 and 11,043.928007 ms.
The scorecard contains every raw value and per-shape outlier set without
discarding them.

Pooled native-phase medians further show that this was a small DiT dispatch
change, not a VAE, transfer, codec, or handoff optimization:

| Component | Before (ms) | After (ms) |
| --- | ---: | ---: |
| UMT5 | 122.681 | 134.808 |
| I2V VAE image encode | 29.072 | 28.9154 |
| **DiT** | **8,766.55** | **8,695.71** |
| VAE video decode | 1,992.66 | 1,988.75 |
| Existing denoise envelope | 10,945.0 | 10,858.0 |
| MP4 encode | 130.405 | 127.296 |
| Native terminal → public first output | 1.185241 | 1.227494 |

## Exact-fidelity rejection

Each geometry produced the same fixed-seed MP4 within an arm, so all five pairs
per shape have the same numeric fidelity values. Baseline media matched the
frozen profile in all 15 pairs. Candidate media and decoded first-frame RGB did
not match baseline in any pair.

| Geometry | Exact paired MP4 | Exact paired first-frame RGB | SSIM | All-frame PSNR (dB) | Temporal-motion ratio | Quality gate |
| --- | --- | --- | ---: | ---: | --- |
| Wide | **FAIL 0/5** | **FAIL 0/5** | 0.970164 | 37.371159 | 1.092606 | **FAIL: exact media/RGB** |
| Tall | **FAIL 0/5** | **FAIL 0/5** | 0.964129 | 35.842211 | 1.076107 | **FAIL: exact media/RGB** |
| Square | **FAIL 0/5** | **FAIL 0/5** | 0.989141 | 43.306101 | 1.023961 | **FAIL: exact media/RGB** |

Candidate MP4 SHA-256 values were respectively
`28ac2f4ee15aa41d621930ac5e97cd0b4683dc20722ae4e7754da9e3b4e01d8f`,
`291611d92ba94e575e50dd23cc24dcb8ce05a7efd4449db544bc14caedc5d479`, and
`7db22a2a98c8fc637e3d3dc0cd59fd794e6a1daab67a23501d9a05e98fe0f6f8`.
The corresponding frozen baseline hashes remain
`6e487e7fe7fca4c439658ccb861431c21f56d7e7619768869517ef115f33e934`,
`dced48d0877e527fd2f224550c827561984d437ebee8694912bcb423b3228a09`, and
`9c01d51996aa1fa6c74657f1177ce84e85165bd52957d42bdcf96bb558a6bfdc`.

The approximate SSIM/PSNR/motion proxies, source-first-frame continuity, nonzero
motion, and decoded geometry controls passed. They do not repair the exact-media
or exact-RGB failures and were not substituted for them. Wide/tall/square source
SHA-256 values remained respectively `477cfafd…3451`, `390465a6…ce27`, and
`c46c343f…d1da`, with exact source bytes reaching native Wan and the required
uniform-scale center-cover geometry preserved.

## Cancellation, malformed input, and recovery

One real non-streaming `/v1/visual/render` disconnect per arm occurred one second
after observed native submit.

| Arm | Disconnect → native abort terminal | Disconnect → client/health settled | Late native records / public media | Final `in_flight` |
| --- | ---: | ---: | ---: | --- |
| Before | 0.588897 ms | 1.567264 ms | 0 / 0 | 0 |
| After | **0.517954 ms** | **1.535566 ms** | **0 / 0** | **0** |

Both arms pass <500 ms. This route exposes no request id, so `/abort` ACK remains
`NOT_APPLICABLE`, not a fabricated measurement. Native terminal dedup passed.
Malformed 500×480 input failed with HTTP 500 and no media in each arm. Immediate
post-cancel retry and post-malformed recovery returned each arm's exact wide MP4
hash: Before `6e487e7f…e934`, After `28ac2f4e…d01d8f`. Every successful request
had one owning native terminal and final public `in_flight=0`.

## Independent verification and raw hashes

The accepted R1 bundle contains **1,776 regular files** and a recursive
self-excluded top-level `SHA256SUMS`; the nested baseline/evidence manifests are
covered. Two independent fresh stdlib-first processes rehashed every entry and
all supplied binaries/model files, independently decoded MessagePack and media,
reconstructed every pair/statistic/outlier, recomputed SSIM/PSNR/motion and exact
media/profile/RGB parity, checked every journal and source/artifact binding, and
recursively recomputed the NOT_PROMOTED decision. Their outputs are
byte-identical. This is **evidence-integrity PASS**, not candidate promotion.

| Record | SHA-256 |
| --- | --- |
| Compact scorecard | `26bc89b838f45f826e55363e25aba4817554644fe783fc5fc6007adc8727c9a0` |
| R1 report | `92c5c8d0d7f8bcfda547c350768aa824e0208e7a49c2b2838115b09f4a06253c` |
| R1 manifest, 1,776 files | `e490a94a7906cfcdf707e7fab85bf1fb535c14c989f2897ddd3b14638c5583da` |
| R1 first and second independent verification | `2b224de1b37ef42852f7ab6330efa6a41a69a080ee40adee07d23f63804ac842` |
| Frozen preregistration | `5be195a5ae7f1b583e409b8be93727c5503644638ea98fa2963bc1c1bd629cf2` |
| Baseline profile report | `73600dc655ecdd938ba61a20177c5db3cd58d3b557eaaa74fa22f0657b803e1e` |
| Baseline profile manifest | `9db5b663a6f0b9655467aeaa53f976b388baf9b4e6c192e0d2109376d213c89a` |

Execution identities include server
`c43934ce6f9ffc4c4218f0930c2038a551737f04f643b2de2c4d107abd98460f`, baseline
Wan `15ed9fd41de4ec635fe0a87b4d3ecd17d2cbd5330934e939d1479b2d28b8adb3`, and
candidate Wan `b3cc811f0030179ec24255911d0b1000c062729e225035cfc633403067ba1c79`.
The final test-only Wan build was
`3b5d81f3d33d4e3a66333730a409a0141a815e0c16a71b96458912a5468607a3`.
The Wan model tree remains
`b118ac07b533cf7d68a4806ec70bdf39041515673e44535afeda5cfefe761616`; model
revision is `UNKNOWN`, with exact file/tree hashes providing identity. FFmpeg and
ffprobe identities are respectively
`ed16af623947494a72e284b6eb8ff225f2da22b38b5d5069c2fd4b4ba3384e41` and
`272f6ebc634a63d9c8b4ca68e964119d980f25154e5aa2c35e5487da48e9a58f`.
The live harness and independent verifier are
`968c4f40440b3446ccddefbaa03d2a3da957fd40ef6f13f9fb4561a818db82de` and
`8f2beb94fe8dba5b9f9454954f52d64f2912ffdfef262dcb5298b2b2f33728d5`.

As a backward-compatibility control, the updated verifier also passed the
complete historical Task #86 bundle (1,566 files) against its original harness
identity and rehashed all supplied binaries/model files, preserving its
NOT_PROMOTED decision. That separate verification is
`b80b82aa6ea3a97609ee668c20f20bb3019823acae91b68f6c1c1ec905fda961`.

## Tests and cleanup

- Candidate and final-runtime `umm-test-local`: PASS. Log SHA-256 values are
  `ad85c4b3c7ecdb5a5c3c88a38d4df24fe64ec8beadc079de76e788c384b34f86` and
  `1a624a1fe08873deefca509c995cfa0248d99cebeb0fccd444db5b1cac5cef74`.
- Frozen protocol/evidence controls: 13/13 PASS; compact scorecard controls:
  4/4 PASS, including exact-media failure and historical boundary2 behavior.
- Candidate native video CTest: 3/3 PASS, including deterministic policy
  boundaries and bitwise synthetic CUDA comparisons. The pre-change red control
  failed to compile because the policy was absent (log
  `032ffbac1f834e09c2e99455220054d934dcbcacd59ba595056c086313514047`); the
  candidate passed green. The subsequent real-weight A/A mismatch remained a
  promotion-blocking red result and was not hidden by the synthetic PASS.
- Final native Wan host/video CTest: 1/1 PASS, log
  `4a7851e417d05ab928d973accf00bd46dd235b2ee8b37fca77c6b179b190fd36`.
- Public native contract-stub completed, cancelled, failed, and recovery controls
  passed with final `in_flight=0`. Narrative source-link and native PCM-join
  self-checks passed; storyboard replay 10/10, raw-SSE replay 11/11, NS-04 replay
  7/7, and native Wan cancellation offline controls 5/5 PASS.
- Final SM90 Stage CTest under `CUDA_VISIBLE_DEVICES=0`: 12/14 PASS, log
  `3f3aa240df73c92692d4766b5e7f475a83a23f429660606e7ec2ce2ca47c6eb2`.
  `stage_common_test` retains the unrelated pre-existing unclosed-marker failure
  documented by Task #86. The image2 allocator test requires two visible devices;
  it was attempted but rejected device 1 and is not counted as a valid PASS under
  the GPU-0-only constraint. Host Stage CTest likewise had 10/11 PASS with the
  same pre-existing stage_common failure. No unrelated runtime fix was made.

Only task-owned processes were stopped. GPU 0 returned to 3 MiB with zero compute
processes and was rechecked at 0% utilization. GPUs 1/7 were not used, and the
resident service/GPUs 2–6 were never contacted, restarted, or deployed to.

## Reproduction

Use exact local assets, an explicitly idle physical GPU 0, and fresh
caller-selected evidence directories. Raw outputs have no tracked absolute path
or default output directory. Build baseline from the Task #86 profile identity,
the historical candidate from `b64f022a3555c925982a9d88274d98c82edddd9f`, and
run the retained harness from `9293fb1aaf8e445ddb493f0d21c6347447bdb293` or
newer with `tests/i2v_geometry_latency_live.py --mode ns02`, the exact-CUDA
preregistration, and the hash-bound baseline profile. Independent verification is
`tests/i2v_geometry_latency_verify.py`; compact scorecard generation is:

```sh
python3 -I tests/ns02_fidelity_speedup_scorecard.py \
  --evidence "<sealed-r1-evidence>" \
  --verification "<fresh-independent-verification.json>" \
  --out "<new-scorecard.json>"
```

The final default runtime intentionally contains no cuBLASLt candidate. No
additional candidate or GPU generation was launched after non-promotion.
