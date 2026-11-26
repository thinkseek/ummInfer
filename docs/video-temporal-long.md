# Long-video temporal-sampling stress fixture and controlled-live contract

## Status and claim boundary

Task #39 added deterministic CPU fixtures, an independent selection oracle, a
controlled-live harness, native scaled-RGB selection fingerprints, and offline
fail-closed tests. Task #44 executed the R4 dual-stack SM90 gate on 2026-11-26
at clean commit `77215cef16e68d4fa2703e25b00dffb5f7a1bd40`:

- `REAL_GENERATION: MEASURED_LIMITED_FIXTURE`
- `LIVE_LATENCY: CONTROLLED_LIVE`
- `PERCEPTUAL_QUALITY: NOT_RUN`
- generalized quality claim: `false`

The technical gate passed all 96 requests and all fail-closed evidence checks.
It did **not** pass a perceptual-quality or generalized-improvement gate. The
explicit expected-substring observations were only 6/24 for default, 6/24 for
uniform, and 0/24 for temporal across both modes; every raw answer is retained
and reproduced below. Earlier R1–R3 setup attempts failed before any model
request and are retained in the R4 evidence bundle.

The synthetic fake endpoint in `tests/long_video_temporal_test.py` is labelled
`SYNTHETIC_FAKE_ENDPOINT_CONTRACT_ONLY`. Its answers and latency values are not
model output or hardware measurements. Expected-substring hits are only raw,
per-case observations and never become a generalized quality claim or a reason
to hide a transport/native-evidence failure.

This extends, rather than replaces, the accepted four-clip short gate in
[video-temporal-live.md](video-temporal-live.md). The long CLI reuses its
request builder, SSE parser, request-keyed native-log parsers, cold-prefix
checks, and temporal-2/MRoPE parity fields.

## Reproducible long-form source bundle

`benchmarks/video-temporal-long/` contains four tracked H.264 MP4 fixtures,
their source/decoded-content records, packet-order evidence, `manifest.json`,
and `SHA256SUMS`. Each clip has 449 frames at 2 fps (224.5 seconds), 96x64 source
RGB, and a 224x160 native analysis/decode target. Generation is single-threaded
`libx264`, `veryslow`, CRF 1, YUV420P, GOP 32, two B-frames, bit-exact container
flags, and no source metadata. The FFmpeg binary and complete version output
are hash-pinned. Encoder drift is rejected rather than silently regenerating
different source bytes.

The cases cover:

| Case | Stress content | Prompt / expected answer scope |
| --- | --- | --- |
| `separated-motion` | separated red/green/yellow motion peaks and long static runs | English; `red` |
| `scene-static` | abrupt cuts at source frames 112/224/336 and long exact-static runs | Chinese; `蓝` |
| `near-duplicate` | low-area near duplicates, exact repeats, a short magenta event | Japanese; one of `magenta`, `マゼンタ`, `紫` |
| `bframe-anchors` | B-frame packet order plus distinct first/last anchors | Spanish; both `blanco` and `negro` |

Every source frame has an independently generated RGB96x64 SHA-256 and an
independently decoded native-resolution RGB SHA-256/FNV-1a record. Packet
evidence proves non-monotonic PTS in decode order alongside monotonic DTS. It
does **not** mislabel decoded presentation order as non-monotonic. The existing
native synthetic candidate test separately covers non-monotonic candidate PTS
without timestamp sorting.

Regenerate only into a new empty directory and compare it byte-for-byte with
the tracked bundle:

```sh
python3 -I python/umminfer_long_video_fixture.py --out /tmp/umm-long-fixture-r1
python3 -I python/umminfer_long_video_fixture.py --check /tmp/umm-long-fixture-r1
python3 -I python/umminfer_long_video_fixture.py --check benchmarks/video-temporal-long
```

## Selection budgets and exact source binding

The independent Python oracle decodes the full source and the native stride-8
candidate set, recomputes the 16x16 Rec.601 integer-luma motion and 16-bin
histogram scene scores, and applies the documented anchors, scene-run
representative, motion ranking, earliest-index tie break, and rounded uniform
fallback. Expected selections are pinned for budgets **4, 8, and 16**.

The public native request contract remains capped at **8 selected frames and
64 candidates**. Thus only budget 8 is legal in the live harness; 4 and 16 are
CPU-oracle coverage. Supplying any other live `--budgets` value fails before
traffic. This does not create a new public frame-budget option.

For these long clips, historical uniform decoding's 0.25-fps floor emits eight
frames from the opening 32 seconds. Independent full-decode content binding
maps them to exact source frames `3,11,19,27,35,43,51,59`; it does not assume
`0,8,16,...`. Temporal mode decodes 57 bounded candidates at source indices
`0,8,...,448` and preserves exact PTS and first/last anchors.

The additive native `selected_frame_content_fnv1a64` diagnostic fingerprints
the exact scaled RGB24 frames entering temporal-2 preprocessing. It does not
alter uniform decode, selected pixels, timestamps, patch packing, or mRoPE. The
harness requires the native sequence to match independent source content in
both arms. Temporal additionally requires exact candidate/source indices and
PTS. Uniform retains its historical nominal native timestamp diagnostics while
the report separately records exact content-bound source indices/timestamps.
FNV-1a is a parity diagnostic, not a security hash; sources and the complete
evidence bundle use SHA-256.

## Controlled serial and batch-four schedule

One invocation requires two separately launched, newly recorded isolated
loopback stacks on two explicitly idle allowed devices:

- serial: batch size 1, client concurrency 1;
- batch-four: batch size 4, client concurrency 4, one four-clip group per arm.

The tracked manifest declares one warmup and three measured repetitions per
arm. `default`, explicit `uniform`, and `temporal` are seeded/interleaved with
seed 390039. Warmups precede measured requests. The full command therefore
issues 48 requests per mode (12 warmup + 36 measured), or 96 total.
Every arm uses the same clip bytes, prompt, model, seed, sampling, token limit,
SSE topology, and selected-frame budget; only `frame_selection` changes.
Default versus explicit-uniform request projection, raw answer, selected
content, temporal-2 packing, and mRoPE must be identical.

Provenance is external to the repository and must pin the new `stack_run_id`,
clean source commit/tree hash, physical GPU, batch size, disabled prefix cache,
live gateway/stage PIDs and binary SHA-256 values, model id/revision, and every
model file's SHA-256. Running `/proc/<pid>/exe` bytes must match those binary
hashes. The serial and batch stacks must have identical binary/model identities.
Model directories with missing or extra unlisted files are rejected. Public
reports retain hashes/PIDs/topology but redact absolute paths.

Before any request, both stacks must pass health and whole-device compute-PID
exclusivity checks. After every step, `in_flight` must be zero and residual
memory within its declared ceiling. Exclusivity is checked again after the
schedule. Source, binaries, complete model file set, and deeply decoded fixture
oracles are revalidated after the run. Missing `prefix_reused=0`, stale bytes,
foreign compute processes, missing logs/artifacts, altered checksums,
selection/temporal-2/mRoPE divergence, incomplete repetitions, or failed
requests fails closed. Whole-device memory samples are kept separate from
request-native allocator telemetry, which this task does not use or modify.

Docker can report NVIDIA compute PIDs in the host namespace while `/proc` and
the running stage use container PIDs. Commit
`77215cef16e68d4fa2703e25b00dffb5f7a1bd40` therefore accepts an additive
`native_stage.compute_pid` alias. The container `pid` still proves the running
`/proc/<pid>/exe` bytes and start identity; `compute_pid` is the sole allowed
NVIDIA-visible process. Missing, malformed, duplicate, or foreign mappings fail
closed. Without the alias, the original single-namespace behavior is unchanged.

### Accepted live command (executed as R4)

R4 launched two fresh stacks from the clean source commit and wrote new
provenance files. It used **explicitly idle physical GPU 0 for serial and GPU 1
for batch-four**. GPU 7 was reserved for another task; GPUs 2–6 and the resident
service were not contacted. The pinned FFmpeg 5.1.9/FFprobe binaries and their
FFmpeg-only compatibility libraries were placed on the isolated stacks' media
path. Then the accepted command ran with these R4 paths:

```sh
UMMINFER_LIVE_OPT_IN=1 python3 -I tests/video_temporal_long_live.py \
  --manifest benchmarks/video-temporal-long/manifest.json \
  --budgets 8 \
  --serial-url http://127.0.0.1:43411 \
  --batch-url http://127.0.0.1:43412 \
  --serial-native-log /tmp/umm-long-task44-r4/serial-stage.log \
  --batch-native-log /tmp/umm-long-task44-r4/batch4-stage.log \
  --serial-provenance /tmp/umm-long-task44-r4/serial-provenance.json \
  --batch-provenance /tmp/umm-long-task44-r4/batch4-provenance.json \
  --source-root "$PWD" \
  --run-id umm-long-sm90-task44-r4 \
  --out-dir /tmp/umm-long-task44-r4/evidence
```

R1–R3 stack PIDs, provenance, native logs, binaries, model manifests, and
evidence paths were not reused. R4 created a fresh clone, build, model manifest,
process set, native logs, and evidence directory.

## R4 SM90 measured evidence — 2026-11-26

### Frozen identity and technical gate

| Identity | R4 value |
| --- | --- |
| Source commit / Git tree | `77215cef16e68d4fa2703e25b00dffb5f7a1bd40` / `a6c8903a172323b07605eccfd1441ae77f385941` |
| Harness source-state SHA-256 | `d1761d15c96c70863c1f6fe559d54f6a296396d753e0c12ea9ea3422253de37b` |
| Gateway SHA-256 | `a18d593d9ade379276ba8ed8c1ac33bbd21ec59ad0632f4d822d6ed9f0b2f77a` |
| Native stage SHA-256 | `06e1b0368ac93948e780d9e776d51d8f167adc935de6aeedc16ff14e9cd8801c` |
| Model / revision | `qwen3-native` Qwen3.8-27B / `UNKNOWN_LOCAL_NO_GIT` |
| Model `config.json` SHA-256 | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| Complete 29-file model manifest SHA-256 | `1dbf82809d9e8f3a0f1897f6eead5623f5752ae8585e8214d28e561076db68df` |
| Fixture manifest SHA-256 | `642ce61e4983cd47d549b0315e0d6f92f01495f936e109487ea2407592a28980` |
| Pinned FFmpeg SHA-256 | `0dafc1360bb07743f76abeb1e4ae16b0aaa1331e9041a0e8d6a8524111dc0cbb` |

Serial used container/native compute PIDs `2352881` / `791130` on physical GPU
0; batch-four used `2352913` / `791163` on physical GPU 1. The gateway PIDs were
`2352890` and `2352922`. All process start identities and `/proc` executable
hashes are in `support/launch-identities.json`. Device exclusivity passed before
and after the schedule with zero foreign compute processes.

The schedule completed **96/96 requests: 24 warmup + 72 measured**, with 12
measured requests per arm per mode. All 96 native records were valid, every
`prefix_reused=0`, and `/health.in_flight=0` after every schedule step. The
independent parity results were:

- default versus explicit uniform: **32/32 PASS** for request projection, raw
  answer bytes, selected content, temporal-2 packing, and MRoPE;
- serial versus batch-four: **48/48 PASS** for requests, answers, and native
  selection/temporal-2/MRoPE;
- within-mode warmup+measured repetitions: **24/24 PASS**.

Independent full-decode FNV source-content binding, exact temporal source
indices/PTS/order, and temporal-2/mRoPE parity passed for every request. This is
a technical selection/transport/reproducibility result, not perceptual
validation.

### Raw answer observations and misses

The following answers were identical in serial and batch-four and across all
three measured repetitions (and the warmup). “Hit” refers only to the manifest's
explicit expected substrings, not to human perceptual quality.

| Case | Default / uniform raw answer | Hit | Temporal raw answer | Hit |
| --- | --- | :---: | --- | :---: |
| `separated-motion` | `green` | no | `green` | no |
| `scene-static` | `蓝色` | yes | `绿色` | no |
| `near-duplicate` | `緑` | no | `緑` | no |
| `bframe-anchors` | `El primer cuadro es de color verde y el último es de color morado.` | no | `El primer cuadro es de color verde y el último es de color blanco.` | no |

Per mode, default and uniform each had **3/12 hits and 9/12 misses**; temporal
had **0/12 hits and 12/12 misses**. Across both modes the counts are 6/24,
6/24, and 0/24 respectively. In particular, temporal selected the independently
verified intended frames but did not produce the expected answers on these four
clips. No temporal-quality improvement, generalized video understanding, or
perceptual-quality claim is made. Raw responses and every repetition remain in
`records/` and `report.json`.

### Latency with actual sample counts

Each row aggregates the 12 valid measured requests across four clips and three
repetitions. Values are milliseconds, median / nearest-rank p95. Batch-four
includes real concurrency-four scheduling/queueing and is not presented as a
serial speedup ratio.

| Mode / arm | n | TTFT median / p95 | Total median / p95 | Native elapsed median / p95 |
| --- | ---: | ---: | ---: | ---: |
| serial / default | 12 | 469.5 / 474.8 | 489.5 / 768.4 | 488.0 / 765.0 |
| serial / uniform | 12 | 467.3 / 474.0 | 487.9 / 767.1 | 486.0 / 764.0 |
| serial / temporal | 12 | 524.7 / 551.1 | 545.4 / 807.9 | 542.5 / 804.0 |
| batch4 / default | 12 | 961.6 / 1419.5 | 1443.6 / 1718.8 | 1441.0 / 1704.0 |
| batch4 / uniform | 12 | 954.3 / 1414.5 | 1435.8 / 1714.3 | 1433.0 / 1700.0 |
| batch4 / temporal | 12 | 1001.7 / 1460.3 | 1482.7 / 1746.7 | 1480.0 / 1732.0 |

Native vision medians were 20.75–21.0 ms and native prefill medians were
282.95–283.35 ms across the six groups, each with n=12. Temporal was slower than
uniform in these measurements; no latency improvement is claimed.

### Whole-device resources, cleanup, and independent verification

| Mode | Schedule steps | NVML samples | Whole-device peak | Final during-service range | Post-stop memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| serial GPU 0 | 12 | 326 | 55,657 MiB | 55,657–55,657 MiB | 3 MiB |
| batch4 GPU 1 | 12 | 250 | 57,861 MiB | 57,783–57,861 MiB | 3 MiB |

These are whole-device NVML observations at 50 ms polling, shared by the four
clips in each schedule step. The declared during-service ceiling was 61,440
MiB. Request-native allocator telemetry was `NOT_USED`, not zero and not
inferred from NVML. All six task-owned processes accepted SIGTERM; no SIGKILL
was needed. Post-cleanup compute-PID lists were empty and both devices returned
to 3 MiB.

The final evidence bundle contains 96 records and 232 `SHA256SUMS` entries;
`SHA256SUMS` SHA-256 is
`354e29c3e98fa39210de45abd1e13b4625bb274cbc46717b9f55e9ff8a14ed45`.
It includes raw SSE/answers/failures, native diagnostics, schedules, prompts,
process identities and clocks, complete model/media/binary/fixture manifests,
all setup failures, native test logs, and the stdlib-independent summary.
`--check-evidence` passed in fresh remote and local processes, independent
aggregate recomputation and parity-count audit matched, and all 232 local `sha256sum -c` checks
passed. R4 also passed `long_video_temporal_test.py` 14/14,
`controlled_measurement_test.py` 38/38, Qwen host/video-input CTest 2/2, and
native-stage CTest 5/5 independently. Evidence remains outside tracked files at
`/tmp/umm-long-task44-r4/evidence` with a locally verified copy at
`/tmp/umm-long-task44-r4-evidence/evidence`.

Budgets 4 and 16 remain CPU-oracle-only. R4 made no live request for them.

## Evidence and independent validation

Each clip/arm/budget/mode/phase/repetition record retains the canonical request
summary, raw answer or bounded failure, response frames, expected-substring
result, native selection/source mapping, cold-prefix execution fields,
first-event/TTFT/total latency, native vision/prefill/decode/elapsed fields, and
shared-step whole-device resource samples. Aggregates report declared,
completed, native-valid, and failed counts plus median, nearest-rank p95, IQR,
MAD, min/max, and explicit metric coverage.

`artifact-manifest.json` and `SHA256SUMS` seal the complete file set. Independent
validation rejects missing, extra, renamed, or altered files and recomputes all
aggregates from raw records:

```sh
python3 -I tests/video_temporal_long_live.py --check-evidence /path/to/evidence
```

Offline checks used for this task:

```sh
python3 -I tests/long_video_temporal_test.py
python3 -I tests/video_temporal_live.py --self-check
python3 -I tests/video_temporal_long_live.py --self-check
python3 -I tests/controlled_measurement_test.py
umm-test-local test
```

## Task #47 R4 root-cause audit and controlled correction boundary — 2026-11-26

### Outcome: fixture/model limitation; no native runtime defect proven

Task #47 started from official main `1136b60` (newer than the requested
`c7e7ed4`), preserved the complete sealed R4 bundle, and used the unchanged
public native stage. The model/data/scoring correction boundary is explicit:
**no native serving implementation, tracked source fixture, expected answer, or
quality-scoring rule was changed**. The accepted R4 outcomes remain 6/24
default, 6/24 uniform, and 0/24 temporal, including every miss.

The established root-cause boundary is a combination of **invalid perceptual
oracle coverage, synthetic color contamination, prompt leakage, and model
limitations on these four fixtures**, not a demonstrated selection/B-frame/
temporal-2/mRoPE implementation defect:

1. Every default/uniform selection ends at source frame 59. Those frames contain
   none of the intended separated motion events (64/176/320), the second scene
   cut (224), the magenta band (288), or the first/last anchors (0/448). Thus the
   uniform expected answers are not justified by the selected long-video pixels.
   In particular, R4's six uniform/default blue hits cannot establish temporal
   understanding: no blue scene was selected. A blue binding tag at source frame
   35 is visible instead.
2. `anchor_tag()` adds unmentioned colored 5x5 source markers to frames 0–63.
   Temporal always selects frame 0, whose green tag survives native scaling with
   138–205 green-dominant pixels depending on the clip. Temporal's intended red,
   blue, magenta, and black/white pixels are also present and correctly ordered,
   but the original responses repeatedly select green. This is a fixture
   contamination and model-grounding limitation, not missing intended frames.
3. The Spanish anchor prompt literally includes `Responde: blanco y negro.`
   Both anchor frames also contain both black and white regions. Its substring
   hits are therefore not clean perceptual evidence.
4. Temporal-2 intentionally packs adjacent selected frames into one channel-major
   slice and uses the pair-first PTS as its text marker. For example, scene-static
   packs red frame 192 with blue frame 224 under the 96.0-second pair-first marker;
   near-duplicate packs static frame 256 with magenta frame 288 under 128.0
   seconds. The oracle must not imply lossless per-frame timestamp semantics.
5. A separate negative probe proves that the native-style FFmpeg command
   zero-normalizes a nonzero initial presentation timestamp. Matroska packets
   retaining 1.5…29.5 seconds decode as 0…28 seconds. All R4 sources start at
   zero, so this boundary does not explain R4. The live uniform control therefore
   uses explicitly named **source-relative** PTS and does not claim absolute-start
   coverage.

### Offline deterministic regressions

`tests/temporal_quality_r4_test.py` contains 15 CPU tests covering the missing
uniform answer anchors, colored-tag contamination, answer leakage, unchanged
multilingual scoring, B-frame packet/decode-order distinction, independent full
mRoPE coordinates, failed-record grouping, nested evidence sealing, exact
rational/container PTS, and lossless RGB round trips. Native host tests add exact
channel-major first/second-frame packing and prove that numeric timestamp text
changes do not alter mRoPE coordinates when token/grid layout is unchanged.

The intervention generator independently decodes the selected source frames and
packages them as lossless RGB H.264. Each derived video must decode to the exact
original eight selected RGB buffers, in the same order, with exact declared
rational packet ticks and native `showinfo` values. Contact sheets and the
nonzero-start negative probe are sealed with the inputs. Derived lossless RGB
controls deliberately contain no B-frames; original B-frame proof comes only
from the original sources and sealed packet/decode evidence.

A host-only `qwen38_prompt_position_audit` executable emits the exact expanded
prompt ids using the frozen tokenizer. It never constructs an Engine or
initializes CUDA. Python independently reconstructs every T/H/W coordinate and
FNV-1a diagnostic digest. All 24 case/arm position entries passed, including
exact agreement with the sealed R4 temporal and uniform baseline digests.

### Controlled serial GPU-0 interventions

The completed run used source commit
`afac5f18c1e71d08cbb2801a0fd6998c8d87dc7b`; the later `5963bd0` correction only
fixed evidence sealing and added no model request. The frozen native model was
Qwen3.8-27B with the same complete 29-file hashes as R4, config SHA-256
`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`.
The gateway SHA-256 was
`337b00db5f3855d6d24372a528459299235222fc6c3714356433686d3d76b1f4`;
the native stage SHA-256 was
`ef4f4e3e5f8c45f0c2e09d9370b052145c62830d611a6aa1e503f360f8db3b61`.
The fixture manifest stayed
`642ce61e4983cd47d549b0315e0d6f92f01495f936e109487ea2407592a28980`;
the R4 `SHA256SUMS` stayed
`354e29c3e98fa39210de45abd1e13b4625bb274cbc46717b9f55e9ff8a14ed45`.

Only explicitly idle physical GPU 0 was used, serial batch size 1, with
`CUDA_VISIBLE_DEVICES=0`, `UMM_QWEN38_PREFIX_CACHE=0`, greedy temperature 0,
`top_p=1`, seed 390039, and 128 maximum output tokens. GPU 7, GPUs 2–6, and the
resident service were not contacted. The schedule used one declared warmup and
three measured repetitions, with seeded interleaving across six arms and all
four scene/motion/static/B-frame cases: 96 requests total, 72 measured.

| Arm | Original expected-substring hits / 12 | Raw answer by case; identical across three measured repetitions |
| --- | ---: | --- |
| temporal pixels + source PTS | 0 | separated=`green`; scene=`绿色`; near=`緑`; anchors=`El primer cuadro es de color verde y el último es de color blanco.` |
| identical temporal pixels + uniform nominal positions | 0 | identical to source-PTS arm on all four cases |
| identical temporal pixels + ordinal normalized PTS | 3 | separated=`green`; scene=`绿色`; near=`緑`; anchors=`blanco y negro` |
| identical uniform pixels + historical nominal positions | 3 | separated=`green`; scene=`蓝色`; near=`緑`; anchors=`El primer cuadro es de color verde y el último es de color morado.` |
| identical uniform pixels + matched source-relative PTS | 3 | identical to uniform-nominal arm on all four cases |
| temporal pixels/source PTS + alternate-language prompt | 0 | separated=`绿色`; scene=`green`; near=` green`; anchors=`The first frame is white and the last frame is black.` |

The language arm deliberately receives **no new scoring aliases**. Consequently
its semantically white/black English answer remains a miss under the original
Spanish substrings. The ordinal arm's three hits occur only on the prompt-leaking
Spanish case and are not evidence of a position fix or generalized improvement.
Source PTS versus uniform positions leaves all four original-prompt answers
unchanged; only ordinal PTS changes the anchor-case answer. No arm establishes
perceptual or generalized quality.

All 96/96 requests completed, all 96 native records passed exact
pixel/order/PTS/temporal-2/media-hash/cold-prefix/independent-position checks,
all 88/88 factor-isolation and within-serial repetition checks passed, and all
32/32 R4-equivalent temporal/uniform baseline checks matched native identity and
raw answers. Existing default-versus-explicit-uniform native tests still pass;
no new default-uniform live comparison was needed because runtime code did not
change. Answer scores never participate in the technical gate.

### Failures retained and evidence verification

Two post-request harness failures are not hidden: an initial report-grouping
`KeyError` after 96 requests, and the final run's initial seal failure caused by
nested R4/source manifest names. Their raw SSE, logs, original reports, invalid
manifests, and corrections are under `failed-attempts/`; the first attempt is not
claimed as a controlled result. Correcting the final seal added zero model
requests. Pre-traffic gateway dependency and media-toolchain failures, plus a
metadata annotation error, are recorded in `support/setup-failures.json`.
No failure or answer miss was removed.

The final local evidence directory is `/tmp/umm-task47-r5-final-evidence`.
Its `SHA256SUMS` SHA-256 is
`060edc180409b8fa15272b416458ffae952905ababceaf95dedd115d7c75b3f0`.
It includes the complete sealed R4 evidence, frozen original/derived inputs,
contact sheets, full independent position vectors, raw SSE, raw native logs,
commands, complete model/binary hashes, aggregates, failures, and cleanup proof.
Fresh remote and local processes passed deep verification:

```sh
python3 -I tests/temporal_quality_r4_live.py \
  --check-evidence /tmp/umm-task47-r5-final-evidence
```

The final native process accepted SIGTERM; GPU 0 returned to 3 MiB with zero
compute processes on both checks. The three stopped container PIDs remained
zombie/reap-pending (`Z`) and are recorded as such, not reported as absent.
`PERCEPTUAL_QUALITY: NOT_RUN`; generalized quality claim: `false`.
