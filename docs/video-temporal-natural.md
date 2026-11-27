# Qwen3.8 natural-footage temporal understanding — Task #72

## Result and claim boundary

The accepted `temporal-natural-r2` gate passed all technical checks. Primary
**temporal-8 scored 42/48, identical to uniform-8 at 42/48**. All 16 paired
case/mode units were unchanged, so the exact two-sided sign-test value is
**1.0**. There are no primary improvements or regressions and no support for a
generalized temporal-selection improvement.

Explicit budget-4 diagnostics were also identical: temporal-4 and uniform-4 both
scored 48/48. This is not a cross-budget significance test and does not establish
that four frames are generally better than eight.

Status:

- `REAL_GENERATION: MEASURED_PREREGISTERED_NATURAL_FIXTURE`
- `LIVE_LATENCY: CONTROLLED_LIVE`
- `NATURAL_VIDEO_FIXTURE_ACCURACY: MEASURED_PREREGISTERED_NATURAL_FIXTURE`
- `PERCEPTUAL_QUALITY: NOT_RUN`
- generalized quality claim: `false`
- selector/runtime change: **none**

Eight short excerpts from one footage compilation and three scene types are not
a representative natural-video corpus. Three deterministic repetitions are not
independent semantic examples, and casefold-substring accuracy is not human or
perceptual quality evaluation.

## Existing natural source and freeze

The source is the existing repository-tracked file
`docs/assets/umminfer-apple-16x9.mp4` at historical commit
`a5d41d39ac6567da53a6ddbb2c4a4534bfdd9104` (`Go minimal: real footage, one line,
one engine`). Its Git blob is `fa078dc543b273922dfc8c07e74f63306778a3ed`; SHA-256
is `81178f8af6c859d908cd6fa4a5348ed9bbe271c071a94a196ff2488c63f0db2e`.
The covering Apache-2.0 license at that commit has SHA-256
`cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` and no
per-asset exception. The excerpts retain this attribution and license.

No procedural replacement, model-generated clip, or unreviewed external download
entered the fixture. Two forest, three whale, and three dandelion excerpts were
visually reviewed before inference. Scene transitions, the rendered planet,
text overlays, and end cards were excluded. Neutral request filenames carry no
semantic labels.

| Clip | Original source frames, inclusive | Frozen frames | Derived MP4 SHA-256 |
| --- | ---: | ---: | --- |
| natural-01 | 0–29 | 30 | `22df061c95db2546973d2c4a7a9a74728b0dd1df05ba7c57bab6e2aefa8d7dd7` |
| natural-02 | 30–59 | 30 | `95208b5e7f34271b149073083687e1f969a79d537c3af76bc1d00f282fc1fa58` |
| natural-03 | 75–94 | 20 | `6b3d7ef01d61df8a6995262b3658f0397b79988786f5c03f5d9c8ce0beefac48` |
| natural-04 | 95–114 | 20 | `86ab1f14ebff6fb8f4825c47c0ac43c1e2619854421de6a8a850d6ea83841015` |
| natural-05 | 115–134 | 20 | `794e5cd3b7b559b37f0cb6f5fc9083bd21cf4a01cd2be05a627778932f75e9b0` |
| natural-06 | 150–164 | 15 | `1e342f8ba8d5fabf1a815e0aa64439217da5ed0a446f350d6c0e2ec978dd0c52` |
| natural-07 | 165–179 | 15 | `dc14c71304aeb6a1cc5e5f086b50b666cb4198b7fbcd35fe41879e332e0028b8` |
| natural-08 | 180–194 | 15 | `cce6f3402a97a96e2ec66b9c8ebbb7b29c9e366d3198a2a4925248cc80de2b90` |

The eight questions cover English, Chinese, Spanish, Japanese, French, German,
Italian, and Portuguese. Every answer is directly derivable from reviewed source
frames. No expected term appears in its own question; no source range, answer,
expected term, or review metadata is sent to the model. The exact multilingual
text and ground-truth derivation are in
`benchmarks/video-temporal-natural/preregistration.json`.

All derived H.264 clips are 256×160 at 30 fps with two B-frames. The full packet
records bind nonmonotonic decode-order PTS, strictly increasing DTS, and
presentation-order RGB/PTS. Temporal selection retains first/last source anchors.
Historical uniform-eight keeps its nominal native timestamps; where exact
repeated source RGB makes a source position ambiguous, every matching source/PTS
candidate is retained rather than claiming uniqueness. Explicit uniform-four
and temporal use exact source indices/PTS.

| Frozen object | SHA-256 |
| --- | --- |
| r2 preregistration | `bec705e56732adab907731da53c139ff953ff184fe0b66df1fb43801ba11e7e2` |
| r2 manifest | `10a7a68244be10ce38565e6ae4f7e97b23a83f1717ea083728e51d4ca44204db` |
| r2 fixture `SHA256SUMS` | `be929f176f7d67fb7fe5764e8880b72e7f1dda6c02f6a5ea36633ad91d069eaa` |
| Task #66 manifest, unchanged | `25c51b6c97cdf9bd6b95991da8a424e8a366ae3105a59099a181d22b65eb2db8` |
| Task #66 preregistration, unchanged | `74eb3eac15233ec5fe508168dfa6b974259544b5b29ef9c28f74ca3539d24bbe` |
| Task #66 fixture `SHA256SUMS`, unchanged | `9a0d0389ac38f48e91dd251dacd6d5e1fd398e01fd7ca52a6705b3585468c000` |

The accepted r2 receipt was written before its model traffic but after the excluded
r1 outputs existed. Its fields explicitly distinguish no prior **r2** output from
`prior_excluded_natural_outputs_seen_at_freeze=true` for r1. Sources, questions,
answers, scoring, population, statistical analysis, and thresholds stayed fixed.
The four Task #66 historical scores and all fixture bytes remain outside the
natural denominator and unchanged.

## Excluded r1 attempt is retained, not accepted

An initial 320-request run used container duration `0.667000` instead of native
stream duration `0.666667` in the independent uniform-eight nominal-PTS oracle
for the 20-frame clips. Fresh verification rejected it. No answer, prompt,
source video, score, alias, or runtime behavior was changed in response. The
entire raw r1 report remains under `excluded-attempts/format-duration-r1/` in the
sealed evidence and is excluded from every accepted result.

| Excluded r1 object | SHA-256 |
| --- | --- |
| report | `a0a3b7d785968790a2d65ba5c6770933a80c267b44437d0b5d04f45590e954f8` |
| `SHA256SUMS` | `e5f82ded3c0670996c5ac51f03e7fc57b80854f9bc8e3349406d913652cbe82c` |

R2 changed only the technical timing oracle and added a regression test. Later
sealing fixed recursive manifests for the nested excluded evidence. Remote
verification normalizes only ffprobe's version-dependent trailing flag padding
(`K_` versus `K__`); raw stored packets and exact PTS/DTS/order remain bound. A
remote full-oracle diagnostic passed all 165 frames before authoritative sealing.
No additional model requests were made for either evidence-only correction.

## Sequential GPU 1 execution and exact native parity

Execution commit was `01885029e4dd43e2035483b9a5dd33b9423a4240`, descending from
newest official main `6e32233f06b8d73677806e6bba0e3ab76a0b12eb`. Its
`native/` plus `src/` diff from that main is empty. Source-state SHA-256 was
`ddcf8d9a794661cdcf652788939709919b409ac630fd3d4a73fde0f348c6ae2d`.

A fresh serial stack ran first, stopped completely, and returned physical GPU 1
to 3 MiB. Only then did a fresh batch-four stack run. GPUs 0 and 7, GPUs 2–6,
and the resident service were never used or contacted. Prefix caching was
disabled; all requests logged `prefix_reused=0`. Sampling was temperature 0,
top-p 1, seed 510072, and at most 64 output tokens.

| Identity | Value |
| --- | --- |
| Model | local Qwen3.8-27B `qwen3-native`, `UNKNOWN_LOCAL_NO_GIT`, 29 hashed files |
| `config.json` | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| Gateway | `ffe97f3f10d22c57b4148df1ccc39b1a772588238b7450b2a5ccf1128ab4f455` |
| Native stage | `9b9d7dbf0677726ca2d276be01ec2114e8b4b67717dcdb7a3c50671260116f0e` |

Each mode has 80 warmup and 160 measured requests. The accepted population is
**320/320 completed and native-valid**, including 240 measured answers. All
**304/304** parity checks pass: 64 default/uniform, 160 serial/batch-four, and 80
within-mode repetition checks. Every default-8/uniform-8 answer and native
record is byte-identical.

The independent verifier fully decodes all 165 source frames and recomputes the
accepted motion-run selector. It checks exact native RGB fingerprints, exact
source indices/PTS where the contract exposes them, all repeated-source
candidates, temporal-2 pair-first timestamps, grid/token identities, mRoPE next
position, and mRoPE digest parity. Every step ended with `in_flight=0`.

Only the six recorded task-owned r2 processes received SIGTERM. No SIGKILL was
used, no other process was stopped, and both post-stop records show zero compute
processes and 3 MiB on GPU 1.

| Mode | NVML samples | Whole-device peak | Post-stop GPU 1 |
| --- | ---: | ---: | --- |
| serial | 812 | 55,657 MiB | 3 MiB; zero compute processes |
| batch-four | 619 | 57,899 MiB | 3 MiB; zero compute processes |

## Quality, uncertainty, and paired transitions

Each arm/budget has eight cases × three repetitions × two modes = 48 measured
answers. Wilson intervals describe request-level proportions only. Sign tests
use 16 case/mode pairs, summing three measured hits per pair.

| Arm / budget | Hits | Accuracy | Wilson 95% |
| --- | ---: | ---: | --- |
| default-8 | 42/48 | 0.875 | [0.7530, 0.9414] |
| uniform-4 | 48/48 | 1.000 | [0.9259, 1.0000] |
| uniform-8 | 42/48 | 0.875 | [0.7530, 0.9414] |
| temporal-4 | 48/48 | 1.000 | [0.9259, 1.0000] |
| **temporal-8** | **42/48** | **0.875** | **[0.7530, 0.9414]** |

| Comparison | Improved / worsened / unchanged pairs | Exact two-sided sign-test p | hit→miss / miss→hit |
| --- | ---: | ---: | ---: |
| primary temporal-8 vs uniform-8 | 0 / 0 / 16 | 1.0 | 0 / 0 |
| diagnostic temporal-4 vs uniform-4 | 0 / 0 / 16 | 1.0 | 0 / 0 |
| temporal-8 vs default-8 | 0 / 0 / 16 | 1.0 | 0 / 0 |

There are no case-level primary/diagnostic improvements or regressions to hide or
reinterpret. The small, saturated sample cannot resolve a generalized selector
advantage.

## Every measured raw answer

All six measured answers in each cell were byte-identical across serial,
batch-four, and repetitions. Warmup answers and every individual request ID are
in the raw evidence and machine-readable summary.

| Case | default-8 | uniform-4 | uniform-8 | temporal-4 | temporal-8 |
| --- | --- | --- | --- | --- | --- |
| natural-01 | `trees` (6/6) | `trees` (6/6) | `trees` (6/6) | `trees` (6/6) | `trees` (6/6) |
| natural-02 | `阳光` (6/6) | `阳光` (6/6) | `阳光` (6/6) | `阳光` (6/6) | `阳光` (6/6) |
| natural-03 | `ballena` (6/6) | `ballena` (6/6) | `ballena` (6/6) | `ballena` (6/6) | `ballena` (6/6) |
| natural-04 | `クジラ` (6/6) | `クジラ` (6/6) | `クジラ` (6/6) | `クジラ` (6/6) | `クジラ` (6/6) |
| natural-05 | `Baleine` (6/6) | `Baleine` (6/6) | `Baleine` (6/6) | `Baleine` (6/6) | `Baleine` (6/6) |
| natural-06 | `Pusteblume` (6/6) | `Pusteblume` (6/6) | `Pusteblume` (6/6) | `Pusteblume` (6/6) | `Pusteblume` (6/6) |
| natural-07 | `Pappina` (0/6) | `Dente di leone` (6/6) | `Pappina` (0/6) | `Dente di leone` (6/6) | `Pappina` (0/6) |
| natural-08 | `sementes` (6/6) | `sementes` (6/6) | `sementes` (6/6) | `sementes` (6/6) | `sementes` (6/6) |

All **18** measured misses are `natural-07` in default-8, uniform-8, or temporal-8:
six per arm, raw answer `Pappina`. The frozen Italian expected terms do not include
that word. No observed-answer alias was added. Four-frame hits are descriptive
outcomes only, not a basis for changing the selector or claiming perceptual gains.

## Latency observations and retained outliers

Medians aggregate 24 measured requests per mode/arm/budget. Batch-four includes
concurrency-four queueing and is not a serial speedup comparison. These are
bounded observations, not a performance-improvement claim.

| Mode / arm | TTFT ms | Total ms | Vision ms | Prefill ms | Native elapsed ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| serial default-8 | 433.009 | 483.602 | 23.4 | 286.35 | 481.0 |
| serial uniform-4 | 377.162 | 431.529 | 17.3 | 219.60 | 427.5 |
| serial uniform-8 | 432.413 | 485.889 | 23.3 | 286.45 | 483.5 |
| serial temporal-4 | 363.571 | 418.573 | 17.2 | 219.45 | 416.0 |
| serial temporal-8 | 440.034 | 489.786 | 23.3 | 286.40 | 487.5 |
| batch-four default-8 | 921.452 | 1445.851 | 23.2 | 286.55 | 1443.5 |
| batch-four uniform-4 | 718.395 | 1159.383 | 17.1 | 220.80 | 1157.0 |
| batch-four uniform-8 | 924.291 | 1446.086 | 23.2 | 286.75 | 1444.0 |
| batch-four temporal-4 | 722.929 | 1161.699 | 17.1 | 220.85 | 1159.5 |
| batch-four temporal-8 | 922.219 | 1455.501 | 23.2 | 286.65 | 1453.0 |

The preregistered inclusive-quartile Tukey rule found **zero** total/native
elapsed outliers. No timing value was removed. Full min/max/p95 values are in the
evidence.

## Independent evidence and verification

The sealed external bundle is `/tmp/umm-task72-evidence`; its archive is
`/tmp/umm-task72-evidence-r2.tar`. It contains all accepted raw answers and SSE
frames, canonical requests, native diagnostics, timing/resource samples,
process/model/source identities, cleanup, preflight history, the excluded r1
bundle, unchanged Task #66 control fixture, and sealed analysis sources.

| Artifact | Result |
| --- | --- |
| Evidence | 1,665 checksummed files, including excluded history |
| Frozen fixture | 34 independently checked files |
| Task #66 regression control | 138 independently checked files |
| Root `SHA256SUMS` | `78ed54498a91553a43c00ca867fc90e1f641355f0e5071fe5fd6a60bdda7ca72` |
| `report.json` | `a2329fc3bc4edb12230777413ed5a2121cf689b43b2bf6b8bb89851163c061ce` |
| Evidence archive | `7466274c08c846f25eeeacae707feb5ced267e1c14b94ae58826f08a876247c7` |
| Independent verifier source | `1740e52067ca1b22f65612c4d128193e0d8be05d076706b3f9c467e041d1fb7a` |
| Local and remote verification receipt, byte-identical | `e187717037316ff2e405f60c7a78c8d4dedbaa284c6ec3d750a45c2a3549b6aa` |

The verifier imports no producer, generator, transport, or scoring module. Fresh
local and remote processes independently reconstructed every answer, all quality
aggregates, Wilson intervals, exact paired sign tests, misses, regressions,
latency summaries, outliers, parity, and checksums:

```sh
python3 -I tests/temporal_natural_verify.py \
  /tmp/umm-task72-evidence --output /tmp/umm-task72-verification-fresh.json
```

Deterministic checks:

- `umm-test-local test`: PASS.
- `python3 -I tests/temporal_natural_test.py`: 17/17 PASS, including deep full
  decode, byte-identical regeneration, repeated-source binding, stream-duration
  precedence, native temporal-2/mRoPE contracts, tamper rejection, and nested
  evidence sealing.
- `python3 -I tests/temporal_heldout_test.py`: 14/14 PASS; Task #66 remains
  unchanged.
- `python3 -I tests/temporal_natural_live.py self-check`: PASS.
- Independent fresh-process verification: PASS locally and remotely.

The machine-readable distillation is
`docs/evidence/video-temporal-natural-sm90.json`. The raw sealed report remains
authoritative for every individual observation.

## Remaining limits

No human/perceptual assessment ran. These eight excerpts do not establish broad
natural-video performance, selector superiority, or cross-budget significance.
No realtime/TTS, narrative-generation, online-training, client, deployment, or
resident-service behavior changed or was validated. The final selector and
serving runtime remain unchanged.
