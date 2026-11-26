# Long-video temporal-sampling stress fixture and controlled-live contract

## Status and claim boundary

This task added deterministic CPU fixtures, an independent selection oracle, a
controlled-live harness, native scaled-RGB selection fingerprints, and offline
fail-closed tests. **No GPU, remote service, deployment, model inference, or real
SM90 run was executed.** In this task:

- `REAL_GENERATION: NOT_RUN`
- `LIVE_LATENCY: NOT_RUN`
- `PERCEPTUAL_QUALITY: NOT_RUN`
- generalized quality claim: `false`

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

### Exact future live command (NOT_RUN here)

First launch the two fresh stacks from the clean source commit and write their
new provenance files. The example requires **explicitly idle physical GPU 0 for
serial and GPU 1 for batch-four**. GPUs 2–6 and the resident service must not be
contacted. Then run exactly:

```sh
UMMINFER_LIVE_OPT_IN=1 python3 -I tests/video_temporal_long_live.py \
  --manifest benchmarks/video-temporal-long/manifest.json \
  --budgets 8 \
  --serial-url http://127.0.0.1:43411 \
  --batch-url http://127.0.0.1:43412 \
  --serial-native-log /tmp/umm-long-r1/serial-stage.log \
  --batch-native-log /tmp/umm-long-r1/batch4-stage.log \
  --serial-provenance /tmp/umm-long-r1/serial-provenance.json \
  --batch-provenance /tmp/umm-long-r1/batch4-provenance.json \
  --source-root "$PWD" \
  --run-id umm-long-sm90-r1 \
  --out-dir /tmp/umm-long-r1/evidence
```

This is a future command, not a recorded run. Do not reuse old stack PIDs,
provenance, native logs, binaries, model manifests, or evidence directories.

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
