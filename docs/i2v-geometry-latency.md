# Public FLUX.2 → native Wan I2V geometry and latency gate (Task #74, 2026-11-27)

## Result and scope

**CONTROLLED_LIVE / PASS** on one explicitly idle NVIDIA Hopper SM90 H20,
physical GPU 1, from official main
`a4320e7678748a2336f74e3c2477c21b30805fb8`.

The real path was public `POST /v1/visual/render` → `UnifiedEngine`/`Coordinator`
→ real FLUX.2 image Stage, followed by the same public route → pure native
C++/CUDA Wan2.2 I2V. The existing CPU echo Stage supplied only legacy entry and
unused unified-VLM readiness; no VLM inference, mock media, TTS, training, or
resident-service request occurred. FLUX exited and GPU 1 was checked idle before
Wan loaded.

The existing uniform-scale center-cover geometry was already correct, so **no
geometry runtime fix was made**. The task added analytic wide/tall/square
regressions, real public evidence, and focused internal phase diagnostics. It also
executed exactly one preregistered latency candidate: direct consumption of the
existing pinned Batched-CFG velocity buffer without its per-step owning-vector
allocation/copy. That candidate is **NOT_PROMOTED**: its tiny mixed-sign effect did
not justify changing the default runtime. Commit
`56ddc30e8079cca6924d0a3d1f9124f4f2ef0f5a` removes the candidate-only API, switch,
fixture, and default call-site change. The final tree retains geometry validation
and phase instrumentation but has **zero candidate-related runtime change**. The
public schema, sampler arithmetic, model precision, TeaCache default, frame count,
and terminal behavior are unchanged. No deployment or final performance
improvement is claimed.

Execution identities:

| Object | Commit |
| --- | --- |
| Official base | `a4320e7678748a2336f74e3c2477c21b30805fb8` |
| Real geometry/profile execution | `198393e311663aea11397cc65f4293e98bfbaa7a` |
| Candidate implementation | `76fe857206840c3aba691af59afb9d27bb7253cf` |
| Real candidate execution | `7cf8519954b345105b05521da39f1994f7aa22e0` |
| Final candidate non-promotion | `56ddc30e8079cca6924d0a3d1f9124f4f2ef0f5a` |

The machine-readable hash/metric summary is
[i2v-geometry-latency-sm90.json](evidence/i2v-geometry-latency-sm90.json). Raw
public/native requests, MessagePack frames, MP4/PNG media, RGB, logs, and the
complete manifests remain outside the tracked tree.

## Real public geometry evidence

All requests used seed 42, 29 frames, 16 fps, 20 steps, guidance 5.0, and
TeaCache 0.0. Every output independently decoded as H.264/yuv420p with the exact
requested dimensions and 29 frames. Wide and tall same-aspect cases require no
crop. The square control explicitly uses one uniform cover scale followed by a
center crop to 704×480.

| Case | Real FLUX source → native target | Source PNG bytes / SHA-256 | Native MP4 bytes / SHA-256 |
| --- | --- | --- | --- |
| Wide | 704×480 → 704×480 | 353,127 / `477cfafdcbd663b3d573d70140c5da355a4874fea920bc721dedc329a8ed3451` | 154,604 / `6e487e7fe7fca4c439658ccb861431c21f56d7e7619768869517ef115f33e934` |
| Tall | 480×704 → 480×704 | 357,689 / `390465a6b7c01e25faaed49e73bb9cf6a49d7e607636988e491805176605ce27` | 163,503 / `dced48d0877e527fd2f224550c827561984d437ebee8694912bcb423b3228a09` |
| Square center-cover | 512×512 → 704×480 | 300,373 / `c46c343fdae91d00a561759c697c53ac7693e39d43e47429a00436ad0067d1da` | 117,921 / `9c01d51996aa1fa6c74657f1177ce84e85165bd52957d42bdcf96bb558a6bfdc` |

The independent verifier decodes each public request's data URL and each tapped
native submit, proving exact source-byte equality. Native completion bytes equal
public response bytes. Native logs independently report the actual decoded
source dimensions and requested output dimensions, so a preceding request's
geometry cannot satisfy the next request. Wide and square sources produce
different wide MP4s under otherwise identical settings, ruling out an ignored
source in this gate.

### First-frame continuity and nonzero motion

The numeric oracle independently center-covers source RGB to the evaluation
aspect ratio and compares it with decoded video frame 0. Motion is frame-0 versus
frame-28 RGB MAE. Thresholds remain first-frame MAE ≤48 and motion MAE ≥1.

| Case | First-frame MAE (all three warm repeats) | Motion MAE (all three) | Warm public complete-artifact median / p95 (ms) |
| --- | --- | --- | --- |
| Wide | 1.962058 / 1.962058 / 1.962058 | 5.305492 / 5.305492 / 5.305492 | 11,159.284371 / 11,175.203703 |
| Tall | 1.936932 / 1.936932 / 1.936932 | 8.516162 / 8.516162 / 8.516162 | 11,127.638951 / 11,163.799575 |
| Square | 1.886932 / 1.886932 / 1.886932 | 4.622601 / 4.622601 / 4.622601 | 11,108.954588 / 11,114.381706 |

The wide, tall, and square warm IQR values were 37.297123, 24.339633, and
29.003964 ms; MAD values were 17.688147, 8.500794, and 6.030131 ms. No IQR
outliers occurred in these nine profile measurements. These are numeric
continuity/motion proxies only. **Perceptual quality, human review, and
perceptual character/style consistency remain NOT_RUN.**

## Deterministic geometry and lifecycle controls

`native/video_tests/test_host.cpp` retains the historical square→wide color-band
case and adds six independent analytic fiducial cases:

- wide→square and tall→square center crops;
- square→wide and square→tall uniform-scale crops;
- wide→wide and tall→tall scaling with no crop.

`R=X+1`, `G=Y+3`, and `B=X+2Y+5` make independent X/Y scaling, transposition,
channel order, and center offsets observable. Expected values are analytic rather
than a copy of the production bilinear loop. Explicit stretch/transposition red
controls cannot pass the oracle, and malformed/null dimensions fail closed.
Request order varies geometry so stale metadata cannot pass.

The public native contract-stub gate separately passed completed, cancelled,
failed, and post-cancel recovery behavior with final `in_flight=0`. Existing
native/replay tests retain exact wrong-hash, tampered-source, missing-anchor,
malformed-dimension, failure, cancellation, and recovery controls. This task did
not repeat Task #67's real cancellation windows. Stub outputs are not real model
evidence.

On the real public path, 500×480 was rejected with HTTP 500 and the exact native
`must be divisible by 32` error, with no media. A following valid request returned
the exact wide-profile MP4 hash. Every successful request had one native terminal;
final public `/health` was `{"status":"ok","in_flight":0}`.

## Measured phase profile

The Stage's internal-only diagnostics decompose image decode, resize, VAE encode,
DiT, VAE decode, and MP4 encode without adding public response fields. The
existing `vae_ms` and `denoise_ms` meanings remain unchanged. Public timing uses
monotonic ticks for submit, response headers, first body observation, and complete
body. The tapped native-terminal tick is an observation at the forwarding tap,
not a claim about an earlier internal CUDA boundary.

Warm pooled medians across all nine measured profile requests:

| Component | Median (ms) |
| --- | ---: |
| Public source preparation (read/hash/base64/JSON) | 6.040476 |
| Native image decode | 7.457890 |
| Native center-cover resize | 5.709580 |
| Wan VAE image encode | 28.347400 |
| UMT5 text encoding | 121.501000 |
| Latent initialization | 8.617120 |
| **Batched-CFG DiT** | **8,811.120000** |
| Wan VAE video decode | 1,993.030000 |
| MP4 encode | 124.284000 |
| Existing native `denoise_ms` envelope | 10,974.400000 |
| Native-terminal observation → public response headers | 1.133328 |
| Public first-artifact-body observation | 11,119.103371 |
| **Public complete artifact** | **11,119.138157** |

DiT was therefore selected as the largest measured component. No historical run
is used as the candidate baseline.

Cold work is separate. Profile FLUX load was 8,015.730 ms; Wan load was
11,579.508 ms. The first wide Wan warmup/cold request was 11,723.377 ms; tall and
square shape warmups were 11,137.117 and 11,138.189 ms. None enters the warm
statistics above. The real public FLUX source requests themselves took
1,508.782 ms wide, 782.947 ms tall, and 646.644 ms square, including their cold
execution context but excluding the separately reported process load.

## Historical candidate execution (NOT_PROMOTED) and exact-parity gate

[preregistration.json](../benchmarks/i2v-geometry-latency/preregistration.json)
was frozen after the profile and before candidate measurements. It binds profile
report SHA-256 `fd31b126ba9147acdd3ca37398d21770b8d1b40ce351a205db39f1ac9948df9e`,
three repetitions, one warmup per shape per fresh Stage block, seeded arm order
(seed 74062), all cold-prefix exclusions, exact source bytes/settings, the primary
metric, and correctness/regression gates.

The historical candidate commit added `WanDit::forward_cfg_pinned`. Its existing
`forward_cfg` kept its owning `std::vector<float>` result. The temporary internal
result view pointed to the already synchronized reusable pinned D2H buffer and was
valid only until the next forward on that `WanDit`. The executed `VideoEngine`
validated its exact latent length and passed it directly to UniPC before another
forward could reuse it. No CUDA kernel or solver arithmetic changed. **None of that
candidate-only API or call-site complexity remains in the final runtime.**

During that historical execution, a direct real-weight SM90 CUDA fixture compared
the legacy owning output and the new view on the same 6,144-value I2V-conditioned
input:
`cfg_pinned_bitwise_equal=1`, legacy vector 24,576 bytes, pinned-result vector
0 bytes. The public gate further required byte-identical MP4 for every same-shape
warmup/measured run, every baseline/candidate pair, and every candidate versus the
verified profile. All passed.

## Controlled latency result

The primary metric is the median paired candidate-minus-baseline public
complete-artifact latency over nine shape/repetition cells. All observations and
outliers are retained.

| Statistic | Baseline | Candidate |
| --- | ---: | ---: |
| Median public complete artifact (ms) | 11,125.887749 | 11,122.020805 |
| p95 (ms) | 11,155.531650 | 11,154.676693 |
| IQR (ms) | 33.034067 | 52.437293 |
| MAD (ms) | 18.947788 | 28.491167 |

| Primary paired result | Value |
| --- | ---: |
| Median candidate−baseline | **−3.687044 ms** |
| Median change relative to baseline median | **−0.0331393%** |
| Paired-delta p95 | 16.259113 ms |
| Paired-delta IQR / MAD | 16.829435 / 7.398493 ms |
| Negative / positive cells | 6 / 9 |
| Retained IQR outliers | −60.663616 and −44.057277 ms |

The native DiT arm medians were 8,816.37 ms baseline and 8,813.14 ms candidate;
existing native `denoise_ms` medians were 10,986.0 and 10,976.3 ms. The result is a
**small measured median reduction under this controlled nine-pair setup**, with
all correctness gates passing. The dispersion and mixed cell signs do not support
a generalized or statistically robust speedup claim. No perceptual-quality,
character/style, deployment, T2V-latency, or cross-host improvement is claimed.

**Promotion decision: NOT_PROMOTED / negative diagnostic evidence.** Avoiding a
24,576-byte fixture copy and observing a −3.687044 ms median on an 11.1 s workload
do not establish a user-perceptible gain: only 6/9 deltas were negative,
paired-delta MAD was 7.398493 ms, and candidate IQR/MAD increased materially.
The raw candidate evidence and exact-parity result remain valid history, but the
final default runtime does not contain the candidate. No final-runtime latency
improvement is claimed.

Candidate model-load times per fresh block ranged from 12,175.013 to
13,128.682 ms. Each block's per-shape warmups and the separate final control block
are retained in the evidence but excluded from warm comparisons. Candidate
malformed-size rejection and exact-hash recovery passed. GPU 1 returned to 3 MiB,
0% utilization, and zero compute processes at closeout.

## Independent verification and retained failed attempts

The verifier imports neither the producer nor its helpers. It implements bounded
MessagePack decoding itself, recursively revalidates the embedded profile in
candidate mode, rehashes every file, reconstructs public/native source and media
bindings, decodes all media/RGB, and recomputes every statistic. Each manifest
covers the complete recursive regular-file set and excludes only its own
top-level `SHA256SUMS`.

| Bundle | Report SHA-256 | Manifest | Remote independent verification |
| --- | --- | --- | --- |
| Profile | `fd31b126ba9147acdd3ca37398d21770b8d1b40ce351a205db39f1ac9948df9e` | 292 files / `92b8094fc24372fc067c0f934289e866fdb7d06f038acf8d68c547e189fd1385` | `07de97d1e6ad1ebc5210e0b5062015012a4030c64a40bf4bf94237620eb0c6cf` |
| Candidate | `5cf3dd18ccd261a722a9bab8bf676b7ebd4127262966260673e94b3a199a928d` | 1,076 files / `cbf1e7ab77a4c5c597906d18c3f1590e58f970251b86090894c5df922f679973` | `c563bb1a417d5b95b9ed1b56e64d344c0e43c3276fdbcad08488c3a8005d234a` |

Fresh local replays also passed with identical report/manifest/result hashes.
Their verification outputs are SHA-256
`57c666c4d25159ba8cb1e2f67359833a9e553d2773e797b9cf193289f37aa84a`
(profile) and
`171050d1c3024d0ac3615065f8b32afe8453545ca6cd7ebcb5c03c8d741f9dc0`
(candidate). Copied-evidence replay explicitly marks execution-binary and model
file rehashing skipped; the remote profile and candidate verifications performed
all supplied binary and model-file identity checks.

Retained corrections/failures, none counted as an initial pass:

1. The first gateway build used an incompatible system cpp-httplib and stopped at
   compilation before launching any model process.
2. The first real attempt generated the three public FLUX sources, then stopped
   before Wan because a post-FLUX NVML sample reported 4% utilization despite
   3 MiB and zero compute processes. A deterministic regression now fails on the
   pre-fix harness and passes with post-owned-stage residual handling. Initial
   admission still requires zero utilization, memory ≤16 MiB, and no processes.
3. Profile generation and producer gates completed before independent
   verification stopped on a verifier-only `stop_ns` versus `stop_monotonic_ns`
   typo. The sealed raw evidence was not changed and no media was regenerated.
   The corrected verifier also fixed and tested its historical-digest comparison.
   Producer-recorded verifier SHA-256
   `50b61708d57ba060339aef9a7483842f46d7d02bec04221dae1f415efcc2aad9`
   remains distinct from corrected remote verifier
   `444de4a18d071779e9f83adfa3736cb55749200ae672021481bf49b17f6f370f`.
4. The candidate used the final verifier without correction. Its recorded and
   current verifier SHA-256 are both
   `95b568ea00dcf77b7409e0066abb480129ceaea3c75146d1ac93fe231158db8f`.

## Reproduction and offline checks

Use an explicitly idle permitted GPU, exact local weights, a fresh evidence
directory, and portable explicit paths. Profile:

```sh
UMMINFER_LIVE_OPT_IN=1 /path/to/venv/bin/python -I \
  tests/i2v_geometry_latency_live.py --mode profile \
  --python /path/to/venv/bin/python \
  --server-binary /path/to/umm-infer \
  --wan-binary /path/to/umm-native-stage-video \
  --flux-stage python/umminfer_flux2_stage.py \
  --flux-model /path/to/FLUX.2-klein-4B \
  --wan-model /path/to/Wan2.2-TI2V-5B-Diffusers \
  --out-dir "<profile-output-dir>" \
  --gpu 1 \
  --base-commit a4320e7678748a2336f74e3c2477c21b30805fb8 \
  --execution-commit 198393e311663aea11397cc65f4293e98bfbaa7a
```

For historical evidence reproduction only, candidate mode requires the exact
profile-instrumented baseline binary and sealed profile; an uninstrumented
historical executable is rejected. Building the executed candidate requires the
preserved `76fe857`/`7cf8519` history, not the final unpromoted tree. No additional
candidate or GPU measurement was launched after the non-promotion decision:

```sh
UMMINFER_LIVE_OPT_IN=1 /path/to/venv/bin/python -I \
  tests/i2v_geometry_latency_live.py --mode candidate \
  --server-binary /path/to/umm-infer \
  --wan-binary /path/to/profile-instrumented-umm-native-stage-video \
  --candidate-wan-binary /path/to/candidate-umm-native-stage-video \
  --wan-model /path/to/Wan2.2-TI2V-5B-Diffusers \
  --baseline-evidence "<profile-output-dir>" \
  --preregistration benchmarks/i2v-geometry-latency/preregistration.json \
  --out-dir "<candidate-output-dir>" \
  --gpu 1 \
  --base-commit a4320e7678748a2336f74e3c2477c21b30805fb8 \
  --execution-commit 7cf8519954b345105b05521da39f1994f7aa22e0
```

Focused checks:

```sh
python3 -I tests/i2v_geometry_latency_test.py
python3 -I tests/i2v_geometry_latency_live.py --self-check
python3 -I tests/native_video_stage_contract_test.py \
  --server-bin /path/to/umm-infer \
  --stage-bin /path/to/umm-native-stage-video --video-backend contract-stub
umm-test-local
```

Results: eight deterministic evidence tests pass; native video host tests 23/23;
local Stage CTest 10/10; public contract-stub
completed/cancelled/failed/recovery PASS; full `umm-test-local` PASS on the final
unpromoted runtime. Remote CUDA Stage CTest 13/13 and the direct pinned fixture are
historical candidate-execution evidence, not tests of a retained candidate API.
A fresh final-tree raw baseline replay also passed with unchanged profile report
`fd31b126…f9e`, manifest `92b8094f…1385`, and verification SHA-256
`57c666c4d25159ba8cb1e2f67359833a9e553d2773e797b9cf193289f37aa84a`.
Real human or
perceptual review remains **NOT_RUN**. Generalized quality/performance, production
deployment, and resident-service validation remain **NOT_RUN/NOT_COVERED** as
applicable.
