# Temporal-clean long-video fixture and controlled-live gate

## Status and evidence boundary

`temporal-clean-r1` is a separate, pre-registered fixture revision addressing the
accepted Task #47 R4 oracle defects. It does not replace or modify the frozen
R4/R5 fixtures, questions, expected answers, scoring, root-cause tooling, native
serving, or online-rollout work.

The completed R1 live gate has the following bounded status:

- `REAL_GENERATION: MEASURED_LIMITED_FIXTURE`
- `LIVE_LATENCY: CONTROLLED_LIVE`
- `FIXTURE_SUBSTRING_ACCURACY: MEASURED_LIMITED_FIXTURE`
- `PERCEPTUAL_QUALITY: NOT_RUN`
- generalized quality claim: `false`

The technical gate passed all 192 requests. The clean quality cases still
produce misses: default and uniform each have 18/24 hits across both modes and
temporal has 12/24. These are model-limitation/fixture-scoring observations,
not a native implementation-defect finding or a generalized quality result.
Casefold expected-substring accuracy is retained only as fixture evidence.
Latency is reported independently from answer outcomes. Observed answers never
created new aliases or changed prompts, anchors, roles, or scoring.

## Frozen preregistration and source binding

The preregistration is `benchmarks/video-temporal-clean/preregistration.json`.
It fixes the source specification, four quality-scored questions, four
selection-miss controls, multilingual expected terms, anchor-visibility metric,
question-after-video request order, seed `510051`, greedy decoding, 64-token
limit, budget 8, one warmup and three measured repetitions, and serial/batch-four
schedule. The receipt `benchmarks/video-temporal-clean.freeze.json` binds:

| Object | SHA-256 |
| --- | --- |
| `preregistration.json` | `96608dc793d54f64f55a7c46862d1b1737cbf807a8807d1b22b2fcd0502d2a77` |
| `manifest.json` | `62f475a74732fa3504850ebe4517dc98903ba39202dce457441c52133a6c96c0` |
| fixture `SHA256SUMS` | `7cbe369ea8dff27da49dca5472f2436a5ea9693bf4c3beeaa3d71a4c18f4616d` |
| freeze receipt itself | `ca9e93ddd80cdc4495e0a7c3893687afb9db31c21d69cc48d87d458aef931dad` |

The live harness refuses traffic unless every frozen fixture byte is present in
`HEAD` and matches the working source. It copies the frozen input and receipt
into the evidence directory before the first request and validates them again
after the schedule.

The four 449-frame / 224.5-second H.264 sources use only shape, binary-pattern,
and luminance semantics. Every source pixel has `R=G=B`; selected decoded frames
also have zero channel spread. Low-area binary-luminance source-binding
barcodes are disjoint from the central answer region. Independent 40x40
geometric anchor cards are not source or decoded crops. The deep oracle checks
NCC and luminance contrast against those cards, exact selected RGB/PTS/order,
static intervals, B-frame packet order, and first/last anchors.

### Quality-scored cases versus selection-miss controls

| Video | Quality-scored question | Anchor requirement in every arm | Separate selection-miss control |
| --- | --- | --- | --- |
| `motion-clean` | English: first moving shape | triangle visible in default, uniform, and temporal | late ring after the middle static interval |
| `scene-clean` | Chinese: shape after the first cut | triangle visible in every arm | late stripes after the third cut |
| `near-clean` | Japanese: symbol retained through long static intervals | ring visible in every arm | late cross during the short change |
| `bframe-clean` | Spanish: opening central symbol | cross visible in every arm | ring in the actual final source frame |

Only the four `quality_scored` questions enter fixture substring accuracy. The
four `selection_miss_control` questions retain every raw answer and hit/miss but
are explicitly excluded from quality accuracy. Their default/uniform selections
must not contain the declared late answer-anchor source frame; temporal must
contain a visible anchor. This prevents missing input evidence from being
relabelled as a model-quality result.

## Deterministic generation and CPU/native validation

Regenerate only into a new empty directory and compare byte-for-byte:

```sh
python3 -I python/umminfer_temporal_clean_fixture.py \
  --out /tmp/umm-temporal-clean-r1-rebuilt \
  --preregistration benchmarks/video-temporal-clean/preregistration.json \
  --freeze /tmp/umm-temporal-clean-r1-rebuilt.freeze.json
python3 -I python/umminfer_temporal_clean_fixture.py \
  --check benchmarks/video-temporal-clean \
  --check-freeze benchmarks/video-temporal-clean.freeze.json
python3 -I tests/temporal_clean_test.py
python3 -I tests/temporal_clean_live.py --self-check
python3 -I tests/long_video_temporal_test.py
```

The source-pixel tests independently prove non-chromatic markers, barcode /
answer-region separation, reference geometry, prompt non-leakage, anchor
visibility, exact-static intervals, B-frame packet versus presentation order,
and original R4 scoring immutability. Synthetic fake-endpoint evidence is always
`SYNTHETIC_FAKE_ENDPOINT_CONTRACT_ONLY`, never real generation, hardware
latency, or quality.

## Controlled-live command

One invocation requires two newly launched isolated loopback stacks built from
the same frozen commit, model files, and binaries: serial on explicitly idle
physical GPU 0 and batch-four on explicitly idle physical GPU 1. GPU 7, GPUs
2–6, and resident port 43301 are forbidden. Only task-owned processes may be
stopped. Budget 8 is the only live-legal budget; 4 and 16 remain CPU-oracle-only.

```sh
UMMINFER_LIVE_OPT_IN=1 python3 -I tests/temporal_clean_live.py \
  --manifest benchmarks/video-temporal-clean/manifest.json \
  --freeze benchmarks/video-temporal-clean.freeze.json \
  --budgets 8 \
  --serial-url http://127.0.0.1:43421 \
  --batch-url http://127.0.0.1:43422 \
  --serial-native-log /tmp/umm-task51-clean-r1/serial-stage.log \
  --batch-native-log /tmp/umm-task51-clean-r1/batch4-stage.log \
  --serial-provenance /tmp/umm-task51-clean-r1/serial-provenance.json \
  --batch-provenance /tmp/umm-task51-clean-r1/batch4-provenance.json \
  --source-root "$PWD" \
  --run-id umm-temporal-clean-task51-r1 \
  --out-dir /tmp/umm-task51-clean-r1/evidence
```

The complete schedule has 192 real requests: eight questions × three arms ×
(one warmup + three measured repetitions) × two modes. Every arm uses identical
video bytes, question, model, seed, decoding, token limit, SSE topology, and
selected-frame budget; only `frame_selection` changes. Default versus explicit
uniform must be identical in request projection, raw answer, selected content,
temporal-2 packing, and mRoPE. Serial/batch-four and repeated-request parity are
also fail-closed technical gates.

The evidence gate requires cold prefixes (`prefix_reused=0`), declared warmup,
device exclusivity, exact source RGB/PTS/order, complete model/binary/source
provenance, `/health.in_flight=0`, bounded residual memory, and complete SHA-256
sealing. Answer misses do not weaken or bypass those gates and do not become a
latency result. Independent fresh-process validation is:

```sh
python3 -I tests/temporal_clean_live.py --check-evidence /path/to/evidence
```

## R1 SM90 measured evidence — 2026-11-26

### Frozen identity and technical gate

The accepted run used source commit
`71c57f36334e6417c28d02f49e0b7a3f64c60f5a` with source-state SHA-256
`c111fe9dc76b2f2cb93ee9ccd9b8ef23ada55ee06372247a6681468d5e759064`.
The fixture was committed before traffic at `eb04792`; the later commit changed
only the preflight identity comparison and added its regression, not any fixture
byte, prompt, answer, score, or native source file.

| Identity | R1 value |
| --- | --- |
| Gateway SHA-256 | `e743c8ed4ec51e82be0ec7e8b100f8ef17360257982aaf34b6bc469bd57bf6ea` |
| Native stage SHA-256 | `88761776e2f39137dffb6178c159db3616eb2a47bb480cef07a92d1ef861be71` |
| Model / revision | Qwen3.8-27B `qwen3-native` / `UNKNOWN_LOCAL_NO_GIT` |
| Model `config.json` SHA-256 | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| Complete model manifest | 29 files, every hash retained in evidence |
| Pinned FFmpeg SHA-256 | `0dafc1360bb07743f76abeb1e4ae16b0aaa1331e9041a0e8d6a8524111dc0cbb` |

Serial used physical GPU 0 with native container/compute PIDs `2890647` /
`2148039`; batch-four used physical GPU 1 with `2890703` / `2148713`. The
running `/proc/<pid>/exe` identities and all six process start identities are
sealed in `support/`. Binary/model/source/fixture identity was revalidated after
the schedule.

The schedule completed **192/192 requests: 48 warmup + 144 measured**. All 192
native records passed exact source RGB, source index/PTS/order, temporal-2,
mRoPE, cold-prefix, and execution checks; every `prefix_reused=0`. The 24
schedule-step cleanup checks and all four pre/post device-exclusivity checks
passed. Independent parity was **208/208**:

- default versus explicit uniform: 64/64 request/answer/selection/temporal-2/
  mRoPE checks;
- serial versus batch-four: 96/96;
- within-mode warmup/measured repetitions: 48/48.

Answer outcomes never participate in these technical gates.

### Quality-scored raw answers and misses

Answers were identical in serial and batch-four and across warmup and all three
measured repetitions. “Hit” means only the preregistered casefold substring
rule. Each per-mode row has 12 measured answers (four cases × three
repetitions); across both modes the denominator is 24.

| Case | Default / uniform raw answer | Hit | Temporal raw answer | Hit |
| --- | --- | :---: | --- | :---: |
| `motion-first-en` | `triangle` | yes | `rectangle` | no |
| `scene-cut-zh` | `三角形` | yes | `三角形` | yes |
| `near-static-ja` | `円` | no | `円` | no |
| `bframe-first-es` | `Cruz` | yes | `cruz` | yes |

Per mode, default and uniform each had **9/12 hits and 3/12 misses**; temporal
had **6/12 hits and 6/12 misses**. Across both modes the counts are **18/24,
18/24, and 12/24**. In particular, temporal selected the independently verified
triangle and ring pixels but answered `rectangle` and `円` on those quality
cases. `円` remains a miss under the frozen Japanese expected terms; it was not
relabelled or added as an alias. No quality improvement or generalized
perceptual claim follows.

### Selection-miss controls, retained separately

These records are not included in any quality-accuracy denominator. Their raw
hits and misses remain available as diagnostics.

| Control | Default / uniform raw answer | Temporal raw answer | Frozen expected terms |
| --- | --- | --- | --- |
| `motion-late-control` | `triangle` | `circle` | `ring` |
| `scene-late-control` | `三角形` | `四条竖线` | `条纹` |
| `near-late-control` | `円` | `プラス` | `cross` or `十字` |
| `bframe-last-control` | `cruz` | `cruz` | `ring` or `anillo` |

Every control answer is a miss: 0/24 for each arm across both modes. Temporal's
`circle`, `プラス`, and `四条竖线` were not converted into new aliases. Its
last-frame control answered `cruz` even though exact native evidence proves the
final ring was selected. These outcomes support a bounded model-limitation
conclusion, not a runtime fix.

### Latency with actual sample counts

Each row independently aggregates 24 measured requests (eight questions × three
repetitions), not only the four quality-scored cases. Values are milliseconds,
median / nearest-rank p95. Batch-four includes real concurrency-four scheduling
and queueing and is not a serial speedup ratio.

| Mode / arm | n | TTFT median / p95 | Total median / p95 | Native elapsed median / p95 |
| --- | ---: | ---: | ---: | ---: |
| serial / default | 24 | 467.6 / 470.9 | 486.5 / 507.6 | 485.0 / 505.0 |
| serial / uniform | 24 | 466.1 / 471.6 | 485.0 / 506.1 | 483.5 / 503.0 |
| serial / temporal | 24 | 519.1 / 524.8 | 541.2 / 573.1 | 538.5 / 571.0 |
| batch4 / default | 24 | 933.0 / 1409.2 | 1116.0 / 1759.1 | 1113.5 / 1751.0 |
| batch4 / uniform | 24 | 933.3 / 1415.9 | 1116.0 / 1765.2 | 1114.0 / 1750.0 |
| batch4 / temporal | 24 | 931.9 / 1456.0 | 1281.3 / 1806.3 | 1278.5 / 1798.0 |

Temporal was slower than uniform in total/native elapsed time. No latency
improvement is claimed. These timings are independent of the answer misses
above.

### Resources, failures, cleanup, and independent verification

| Mode | Measured NVML samples | Whole-device peak | Final during-service range | Post-stop memory |
| --- | ---: | ---: | ---: | ---: |
| serial GPU 0 | 135 default/uniform; 147 temporal | 55,653 MiB | 55,653–55,653 MiB | 3 MiB |
| batch4 GPU 1 | 96 default; 95 uniform; 94 temporal | 57,859 MiB | 57,819–57,859 MiB | 3 MiB |

Whole-device samples were collected at 50 ms and are shared by the eight cases
in each schedule step. Request-native allocator telemetry was `NOT_USED`.
All six task-owned processes accepted SIGTERM and no SIGKILL was needed. Their
container states remained zombie/reap-pending (`Z`), recorded explicitly rather
than reported as absent. Both devices had zero compute processes after cleanup.

Three pre-traffic/test issues are preserved in `support/setup-failures.json`
and `failed-attempts/`: the local timing-sensitive nucleus failure followed by
a serial 2/2 PASS, the remote cross-host compressed-byte rebuild mismatch with
the exact frozen bundle's deep oracle still passing, and the zero-request
`compute_used_mib` identity-preflight failure followed by a focused harness
regression and fresh stacks. No assertion, fixture, prompt, expected answer, or
runtime was changed to obtain the live result.

The local evidence copy is `/tmp/umm-task51-clean-r1-evidence`. It contains 478
files, including every raw answer/SSE record, miss, native diagnostic, frozen
input, model/binary/source identity, native test result, setup failure, and
cleanup record. `SHA256SUMS` SHA-256 is
`536a44d0781b581bd4a63cc6c5a8b61481c8d7bba937e9baf1ea82475d3bb790`.
Fresh remote and local `--check-evidence` processes passed deep fixture and
aggregate/parity recomputation, and independent `sha256sum -c` passed all
entries. No native serving or online-rollout file was modified.
