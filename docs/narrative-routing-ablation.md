# Stage C narrative-routing and visual-form ablations

## Ablation 1: narrative routing

This harness compares only the four routing baselines that already exist in the
public narrative contract. It does not implement a learned router, a new serving
path, a quality evaluator, or an optimizer.

| Config value | Existing request behavior |
| --- | --- |
| `text_only` | `narrative.enabled=false`; modalities `text,audio` |
| `forced_image` | `narrative.enabled=true`, `visual_mode=image`; modalities `text,audio,image` |
| `forced_video` | `narrative.enabled=true`, `visual_mode=video`; modalities `text,audio,video` |
| `auto_tag` | `narrative.enabled=true`, `visual_mode=auto`; modalities `text,audio,image,video` |

The current engine's auto-tag parser and no-tag fallback are used unchanged.
Forced image/video therefore exercise the existing `visual_mode` behavior,
including fallback, rather than a new forcing implementation.

## Versioned inputs

- `benchmarks/ablations/narrative-routing.json`: ablation schema v1; requires
  trajectory schema v2 and exactly the four policies above.
- `benchmarks/ablations/narrative-routing-prompts.json`: dataset schema v1.
  The checked-in dataset is deliberately **UNLABELLED** contract data.
- `benchmarks/ablations/narrative-routing-replay.json`: replay schema v1, with
  eight canned SSE responses and recorded contract latency observations. Its
  media and timings are not real model or hardware evidence.
- `benchmarks/ablations/narrative-routing-sm90.json` and
  `narrative-routing-sm90-prompts.json`: the two-prompt, human-labelled live
  diagnostic configuration. Labels are the intended visual modality multiset,
  not a validated quality benchmark. The model revision records the local
  `config.json` SHA-256. `weights_sha256` is the SHA-256 of the newline-terminated
  `sha256sum model-*.safetensors` manifest (18 shards, relative filenames).
  Audio is absent from every SM90 arm as a fixed condition, so its TTFA coverage
  is `NOT_APPLICABLE`; the offline routing replay retains its audio contract.
- `benchmarks/ablations/narrative-routing-controlled-sm90.json`: the pinned
  two-GPU controlled live manifest using that labelled dataset, one warmup and
  three measured repetitions per arm, seeded interleaving, real-media decode,
  and complete language/image/video/ASR provenance. Its allowed host PIDs bind
  the one recorded isolated execution and intentionally fail on any later stack;
  rerun by recording a new isolated stack, never by widening the allowlist.

The config pins the dataset and replay byte SHA-256. Every replay case also pins
its full request hash, fixed-request hash, and raw response hash. A config,
prompt, sampling, visual-setting, response, or policy edit that is not matched by
the corresponding hash fails closed.

## Single-variable rule

`build_request()` constructs every policy request from the same model, messages,
sampling, audio, image, and video values. Only `modalities`, `narrative`, and the
presence of the `audio`/`image`/`video` option objects are overlaid. Presence is
part of routing because the public parser treats a present options object as
enabling that output even when `modalities` omits it. An option object's value is
byte-identical whenever selected; a disabled modality's object is absent.

`fixed_request_sha256` hashes the complete request except those five activation
fields. The report contains one such hash per prompt and a `fixed_matrix_sha256`
over all prompts; these values must agree across policies. Tests explicitly
assert that a disabled modality has no options object, preventing accidental
all-modality fan-out.

The checked-in fixed values include seed 42, temperature 0, top-p 1, 256 maximum
tokens, the same streaming-audio settings, two visual slots, and identical
image/video sizes, steps, guidance, and seeds whenever those modalities are
selected.

## Ablation 2: visual form

`benchmarks/ablations/visual-form.json` selects `variable=visual_form` and exactly
three forms:

| Form | First-stage public request | Second public request |
| --- | --- | --- |
| `image_only` | `narrative.visual_mode=image`; modalities `text,image` | none |
| `video_only` | `narrative.visual_mode=video`; modalities `text,video` | none |
| `image_then_i2v_animate` | byte-identical to `image_only` | `POST /v1/visual/render` with `kind=video` and the generated PNG |

`benchmarks/ablations/visual-form-replay.json` contains six filtered canned SSE
responses and two canned raw MP4 render responses. These are contract fixtures,
not real image/video evidence.

The model, messages, sampling, image/video options, seeds, and maximum slot count
stay fixed. There is no audio leg in this ablation. `fixed_request_sha256` and
`fixed_matrix_sha256` cover the same fixed request projection as Ablation 1. The
only additional change is the explicitly reported `post_processing.i2v_animate`
step. Image-only and I2V must have identical `request_sha256`; the test suite
fails otherwise.

Each animation records its source slot and image artifact SHA-256, complete
render-request hash, fixed render-request hash (all fields except `image`), HTTP
and terminal status, monotonic/replayed latency, and output MP4 SHA-256. Derived
animations live in out-of-band `record.ablation.animations` and case reports, not
in invented chat SSE events. Latency is omitted from the exported animation
metadata, preserving live-versus-replay trajectory byte identity. Every source
and derived artifact remains in CAS and is covered by `SHA256SUMS`.

Visual-form completion and animation success measure protocol/artifact
completion only. Image-text relevance, perceptual quality, and user blind
preference remain `NOT_COVERED`; no numerical visual-quality claim is made by the
harness.

`benchmarks/ablations/visual-form-sm90.json` is the live real-model manifest. It
pins Qwen3.8-27B, native Qwen-Image 2.1, and native Wan2.2-TI2V-5B configuration
hashes and the SHA-256 of each model's relative-filename Safetensors manifest.
These manifest hashes identify the available local bytes; they are not claimed
to be upstream revision IDs.

`benchmarks/ablations/visual-form-controlled-sm90.json` is the distinct pinned
single-H20 controlled manifest. It uses the human-labelled two-prompt dataset,
one warmup and three measured repetitions per arm, seeded interleaving, exactly
GPU 0 with three recorded compute PIDs, real-media decode, per-step resource
polling, and complete language/image/video executable and available-weight
hashes. Its PIDs intentionally bind the one recorded isolated stack; a rerun
requires a new isolated stack and config, never an expanded allowlist.

With `validation.decode_media=true`, every real PNG is CRC-checked, zlib
decoded, and unfiltered to RGB8; every real MP4 is fully decoded by ffmpeg and
counted by ffprobe. Dimensions and frame count must match the request. Contract
replay keeps media decoding `NOT_COVERED` because its deliberately tiny PNG/MP4
markers are not real media evidence.

The live manifest also enables `i2v_first_frame_crop`. The runner decodes the
source PNG and first H.264 frame, mirrors the native engine's half-pixel bilinear
aspect-preserving center crop, and compares the frame against center-crop,
stretch, and top-left-crop references. The behavioral gate requires center-crop
MAE no greater than 32.0 and no worse than either alternative. PSNR, MAE, frame
hashes, and pass/fail are reported as **algorithmic behavior proxies**, not human
perceptual-quality scores. Source and derived artifacts remain CAS-bound.

For live resource observations, repeat `--gpu-index` for each explicitly idle,
assigned GPU. The runner polls only those devices with `nvidia-smi` and reports
per-arm baseline, peak, final MiB, sample count, and poll interval. These are
whole-device used-memory samples that include the persistent model baseline, not
request-only or allocator-native peaks. Without the flags, VRAM coverage remains
`NOT_COVERED`. Reports also aggregate terminal states, terminal leg states,
animation states, observed video frame counts, and animation terminal latency.

## Run

CPU-only deterministic replay:

```text
python3 -I python/umminfer_ablation_run.py \
  --config benchmarks/ablations/narrative-routing.json \
  --replay-only --out /tmp/umminfer-narrative-routing-ablation
python3 -I python/umminfer_ablation_run.py \
  --config benchmarks/ablations/visual-form.json \
  --replay-only --out /tmp/umminfer-visual-form-ablation
```

Live recording against an explicitly selected public HTTP endpoint:

```text
python3 -I python/umminfer_ablation_run.py \
  --config benchmarks/ablations/narrative-routing.json \
  --base-url http://127.0.0.1:PORT \
  --out /tmp/umminfer-narrative-routing-live
python3 -I python/umminfer_ablation_run.py \
  --config benchmarks/ablations/visual-form-sm90.json \
  --base-url http://127.0.0.1:PORT \
  --gpu-index 0 --gpu-index 1 --gpu-index 7 \
  --timeout-s 900 --out /tmp/umminfer-visual-form-sm90-live
```

Live mode uses the existing out-of-process schema-v2 recorder. An optional
normalized-event observer measures monotonic latency outside the exported
record. No C++ serving file imports the runner, metrics, reward, trainer, or
optimizer. Every report carries one of three evidence classes:
`CONTRACT_REPLAY` for deterministic replay, `UNCONTROLLED_LIVE` for ad hoc
live observation, and `CONTROLLED_LIVE` only for a live run that satisfies the
controlled measurement contract below. `claims.latency_evidence` mirrors the
same token. For real model provenance, set a non-null `model.weights_sha256`;
the default contract model has no weights and reports weights coverage as
`NOT_COVERED`. `model.config_sha256` hashes the runner's `{id,revision}`
identity descriptor, not an external model card or weight file.

The output directory must be absent or empty. It contains:

```text
report.json
SHA256SUMS
<text_only|forced_image|forced_video|auto_tag>/
  report.json
  trajectory.jsonl
  artifacts/
# or, for visual form:
<image_only|video_only|image_then_i2v_animate>/
  report.json
  trajectory.jsonl
  artifacts/
# additionally, only for live measurement-contract runs:
warmup/<arm>/
  trajectory.jsonl
  artifacts/
```

Each trajectory is validated by the existing independent RL worker after CAS
reassembly and reward annotation. Media bytes stay out of JSONL. `SHA256SUMS`
covers every report, trajectory, and artifact. Two replay runs from the same
commit and inputs produce byte-identical relative file trees; the regression
test compares the complete trees, not just summary values.

## Controlled measurement contract

`python/umminfer_controlled_measurement.py` replaces ad hoc
`UNCONTROLLED_LIVE` inputs with a reusable, declarative experiment contract.
An optional top-level `measurement` object (schema v1) in any ablation config
declares:

- `warmup_repetitions` (>= 0) and `measured_repetitions` (>= 1): declared per
  arm counts. Warmups run first and are exported out-of-band under `warmup/`;
  they never enter measured trajectories, metrics, or run status.
- `arm_order`: `interleaved` (repetition-major round-robin, every arm once per
  repetition) or `randomized` (full shuffle of the measured arm x repetition
  matrix). Both orders are a SHA-256-ordered permutation keyed by the recorded
  `seed`, identical on every host and Python version; no wall-clock or PRNG
  state leaks into the schedule. One schedule step executes every prompt of
  one arm once, in fixed dataset order.
- `exclusive_device`: `required`, declared `gpu_indexes`, `allowed_pids` (the
  serving stack under test), `max_other_compute_processes`, and an optional
  `max_memory_used_mib` baseline ceiling. The precondition queries nvidia-smi
  compute apps on exactly the declared devices; any foreign process fails it.
- `cleanup`: `require_in_flight_zero` (polled against the existing public
  `/health` interface after every step), an optional
  `max_residual_memory_mib` tolerance against the recorded baseline, and a
  poll `timeout_s`. `cooldown_ms` declares an optional inter-step cooldown.

`benchmarks/ablations/narrative-routing-controlled-example.json` is a complete
checked-in example. The same block is present in
`visual-form-controlled-sm90.json` for the scheduled live gate; request
construction, single-variable overlays, media validation, and the I2V
first-stage equivalence gate are unchanged, and every repetition of one
(arm, prompt) sends a byte-identical request.

Replay with a `measurement` block stays `CONTRACT_REPLAY`: the report embeds
the validated schedule and its `schedule_sha256` with `status=NOT_RUN`, and
two replay runs remain byte-identical. Live mode executes the schedule and
adds a `measurement` report section with the contract, schedule, precondition
and cleanup records, per-arm warmup/measured coverage, warmup summaries,
per-step whole-device resource rows, and the eligibility verdict.

`CONTROLLED_LIVE` is refused, with explicit `eligibility.reasons`, whenever
any dimension is missing: declared warmups not executed, measured repetitions
incomplete, any measured request not completed, terminal latency absent, any
terminal leg not completed or any leg error, exclusivity not declared or the
precondition failing or never executed, any cleanup check failing, or any
source/config/data/artifact hash not pinned. The language model, both declared
`visual_backends` (`image` and `video`), and every optional named
`auxiliary_backends` entry must each pin a model config-file hash, the
available-weights manifest hash, and the backend executable `binary_sha256`.
The contract example intentionally lacks that real provenance and has
`weights_sha256: null`, so its live runs always downgrade.
`--require-controlled` exits 2 unless the run earns `CONTROLLED_LIVE`; a
failing precondition then aborts before any request is sent.

Aggregation reports median, nearest-rank p95, IQR, and MAD dispersion with
explicit `sample_count`/`total_count` per prompt across repetitions and pooled
per arm, alongside the existing per-case latency fields. Measured cases and
trajectory metadata carry `repetition` and `schedule_index`; visual-form
first-stage image equivalence is checked per repetition. A missing pair,
request-hash mismatch, or source-PNG byte mismatch both fails the combined run
and revokes `CONTROLLED_LIVE`; the strict flag cannot hide that failure.

Telemetry stays within existing interfaces: `/health` `in_flight` for cleanup
and nvidia-smi whole-device used-memory for baselines, residuals, and per-arm
resource summaries. Every controlled schedule step has a 50 ms poll monitor,
so its recorded peak is the maximum observed during that step rather than a
maximum of end-of-step values; per-arm sample counts sum those actual polls.
No existing public interface exposes request-scoped native allocator/VRAM
telemetry, so `request_scoped_telemetry.coverage` is honestly `NOT_COVERED`
and every device number is labelled
`whole-device-used-memory`, including persistent model baselines and, under
interleaving, other arms' activity. No serving, realtime/VAD, CLI realtime, or
post-training file is modified by this layer, and no Wave 16 kernel, showcase,
routing, or joint-trainer code is duplicated.

Before the accepted run documented below, controlled coverage consisted only of
deterministic replay and loopback canned-response tests with injected samplers;
those tests were never real controlled performance evidence. The real gate was
then executed as follows, and any future rerun must use the same procedure:

1. pin the language model, image/video backends, and any required auxiliary
   stages with config, available-weight manifest, and executable hashes in a
   controlled live config on a clean committed tree;
2. start the isolated stack on declared idle GPUs, record its serving pids in
   `exclusive_device.allowed_pids`, and keep the resident service untouched;
3. run `python3 -I python/umminfer_ablation_run.py --config <config>
   --base-url http://127.0.0.1:PORT --require-controlled --out <empty-dir>`,
   optionally with `--gpu-index` for additional whole-device sampling;
4. publish the report, `SHA256SUMS`, schedule hash, and eligibility verdict;
   any refusal reason downgrades the run to `UNCONTROLLED_LIVE` and it must
   not be cited as controlled performance evidence.

## Latency and status fields

Every case reports:

- `latency.ttft_ms`: first non-empty VLM text delta;
- `latency.ttfa_ms`: first audio artifact chunk;
- `latency.terminal_ms`: normalized terminal event (or recorder return when a
  synthetic terminal is required);
- per-slot `dispatch_ms`, `first_artifact_ms`, `slot_to_artifact_ms`, and
  `settlement_ms`;
- per-I2V-render `latency.terminal_ms` (visual-form ablation only);
- `terminal_state`, `error`, `terminal_legs`, and `leg_errors`;
- complete output descriptors and SHA-256 for every published artifact.

Latency is represented as integer microseconds at the observation boundary and
serialized as milliseconds. Replay uses the recorded `latency_us` observations
and marks them `recorded-contract-observation`; live mode uses
`monotonic-live`. Neither latency source enters trajectory schema v2, preserving
its byte determinism. Policy aggregates report median and nearest-rank p95 with
sample/total counts. Missing values remain null and reduce coverage, including
terminal latency. Completed zero-slot cases (including auto-tag abstention) have
slot latency `NOT_APPLICABLE`, not zero or missing coverage.

## Labels, routing metrics, and reward

An optional prompt field supplies labels without changing the request:

```json
"expected_visual_modalities": ["image", "video"]
```

The runner copies this field into out-of-band `record.ablation` analysis
metadata. Predictions come only from observed schema-v2 `visual_slots`, never
from the requested policy. With labels:

- routing F1 is micro-averaged over the image/video modality sets;
- a redundant slot is a predicted slot that cannot be matched to the labelled
  modality multiset, and redundancy is redundant slots / predicted visual slots;
- the diagnostic export-only reward is `F1 - redundant_slot_rate`;
- a labelled empty prediction with an empty label has F1 1 and reward 1.

Without labels, per-case and aggregate F1/redundancy stay null with
`NOT_COVERED`, and reward score stays JSON null. A partially labelled matrix is
`PARTIAL` and aggregates only labelled cases. This reward is a transparent
ablation diagnostic, not a validated business objective or evidence of training.

## Provenance and claim boundary

Combined and per-policy reports record the current source commit, raw config and
dataset hashes, replay hash (when configured), an artifact-manifest hash, model
identity/config and weights hashes, trajectory hashes, and the fixed-matrix
hash. `source_state_sha256` additionally binds the binary tracked diff from
`HEAD` and every untracked, non-ignored source byte; `source_tree_dirty` makes
uncommitted execution explicit. Publishable evidence should still come from a
clean committed tree. Metric coverage is explicit per case, per policy, and in
the combined report.

The checked-in replay reports evidence class `CONTRACT_REPLAY`, learned routing
`NOT_RUN`, optimizer `NOT_RUN`, quality improvement `NOT_COVERED`, and real model
quality `NOT_RUN`. No latency, F1, redundancy, quality, learned-routing, or
optimizer improvement claim may be made from these canned fixtures. Live runs
without a measurement contract are `UNCONTROLLED_LIVE`; `CONTROLLED_LIVE` is an
eligibility verdict, never a flag, and the historical accepted SM90 sections
below keep their original `UNCONTROLLED_LIVE_MEASUREMENT` wording unchanged.

## Controlled SM90 narrative-routing result — 2026-11-26

The remaining real Stage C routing gate ran on the integrated Wave 17 tree.
Base `b7a3e68abaed23ec3340385a1f8645a91870c862` plus controlled integration
`4ddf56a55a55884b873db4287b10f4dfa922c40f` produced the clean measurement
source `9dab36391259a5663c0f2ced7d37db47eef93d5a`. The combined verdict is
**CONTROLLED_LIVE / COMPLETED**, with no eligibility refusal reasons.

Only explicitly idle physical GPUs 0 and 1 were assigned. ASR (required entry,
but unused by every request) and Qwen3.8-27B ran on GPU 0; native Qwen-Image
2.1 and native C++/CUDA Wan2.2 ran on GPU 1. The isolated loopback endpoint was
port 43420. No TTS worker or audio request existed. GPU 7, GPUs 2–6, and the
resident service on port 43301 were not assigned, contacted, restarted, or
used. A rejected non-stream narrative initialization attempt and two successful
non-stream image/video initialization requests were explicitly excluded from
both warmup and measurement; they only forced lazy model/workspace allocation
before the baseline.

The schedule contained 16 arm steps: four warmup steps followed by 12 measured
steps in seeded, repetition-major interleaved order. Each step executed both
pinned labelled prompts. Thus every arm completed two warmup requests and six
measured requests, for 8 excluded warmup requests and 24 measured requests
overall. Every measured arm/prompt pair completed repetitions 1, 2, and 3 with
a byte-identical request. All 16 cleanup checks reported `in_flight=0` and
residual memory within 128 MiB of the fully initialized baseline.

### Controlled routing and latency measurements

F1 is micro-averaged over six measured cases per arm; redundancy is redundant
slots divided by predicted visual slots. Pooled latency statistics have six
samples per arm. p95 is nearest-rank (therefore the maximum at n=6), IQR and
MAD are milliseconds of dispersion, and every cell below has full 6/6 coverage.
The JSON report additionally provides each prompt separately with 3/3 samples.
These are controlled measurements for this exact stack and two-prompt labelled
diagnostic, not a generalized quality benchmark.

| Policy | Completed | Micro F1 | Redundant-slot rate | TTFT median | TTFT p95 | TTFT IQR | TTFT MAD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `text_only` | 6/6 | 0.000 | 0.000 | 303.937 | 306.170 | 4.243 | 2.122 |
| `forced_image` | 6/6 | 0.800 | 0.000 | 362.441 | 365.343 | 5.323 | 2.662 |
| `forced_video` | 6/6 | 0.400 | 0.500 | 363.461 | 366.277 | 5.590 | 2.768 |
| `auto_tag` | 6/6 | 0.800 | 0.000 | 512.472 | 515.060 | 4.843 | 2.422 |

| Policy | Terminal median ms | Terminal p95 ms | Terminal IQR ms | Terminal MAD ms | Slot-to-artifact median / p95 ms | Slot n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `text_only` | 7,640.131 | 7,677.264 | 72.433 | 36.217 | N/A | 0 |
| `forced_image` | 12,575.509 | 12,664.602 | 200.329 | 87.078 | 8,662.356 / 8,683.964 | 6/6 |
| `forced_video` | 8,616.317 | 8,656.097 | 79.199 | 39.600 | 3,008.874 / 3,176.986 | 6/6 |
| `auto_tag` | 17,110.540 | 19,792.038 | 5,314.602 | 2,657.301 | 9,363.917 / 10,128.760 | 6/6 |

Auto-tag emitted one image for each prompt in every repetition: it matched the
conservation label but missed the lantern video label, yielding F1 0.800. This
is intentionally not reconciled with or replaced by the earlier uncontrolled
F1 1.000 result below. Forced image missed the lantern video label; forced
video missed its image label and emitted a redundant conservation video.
Pooled auto-tag terminal dispersion spans the two prompts' different visual
work; use the per-prompt 3-sample summaries before interpreting dispersion.
TTFA is `NOT_APPLICABLE` in every arm. Learned routing and optimizer runs remain
`NOT_RUN`; perceptual quality, generalized routing quality, and quality
improvement remain unmeasured.

### Controlled artifacts, provenance, and cleanup

All terminal states were `completed`, all 18 measured visual legs were
`completed`, and there were no terminal leg errors, failed cases, or cancelled
cases. Text-only correctly had no visual legs. Every PNG passed full RGB8
decode at 512x512; every MP4 passed full ffmpeg/ffprobe decode as H.264,
480x320, 16 fps, 13 frames. The three repetitions of each visual arm/prompt
pair produced identical artifact hashes. Six unique measured artifacts were:

| Policy / prompt | Artifact | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| forced image / lantern | PNG | 98,594 | `ea3534e9e2ef358a2973123b14f9f143a06a4eb8fbe0f28a058a2e9733e9eee9` |
| forced image / conservation | PNG | 269,919 | `5f753197ca1cbdec6cb4a68fe7a3d81d8e269a85d9a1c65adf360b8f9966d6b9` |
| forced video / lantern | MP4 | 65,731 | `6c1def132fd7bed82fa4c99560b27b85c8a714d861d425d3e6850c5f7207bde1` |
| forced video / conservation | MP4 | 67,410 | `a4fd62621ff0dcf66e9a608fafa7c6cde046e425ff9ce46bfaebbbb25300d864` |
| auto-tag / lantern | PNG | 158,032 | `af53567b9db94523fd51364420dc55d43a3cc71bc4c7399dd381f2a8b0256364` |
| auto-tag / conservation | PNG | 81,840 | `8d33f94165159efbe997fcbcc3a88f9c29515f09547c63ed2d9ef47e1ce6797d` |

An independent checker revalidated the schedule hash, all repetition and
terminal/leg coverage, no-audio request overlays, model/backend provenance, and
all 25 files covered by `SHA256SUMS`; a separate `sha256sum -c SHA256SUMS` also
passed 25/25. All 36 available Safetensors files across Qwen3.8 (18), image
(7), Wan2.2 (9), and ASR (2) passed their recorded relative-path manifests.

| Evidence item | SHA-256 |
| --- | --- |
| Combined `report.json` | `bdf7ec148124440518fc5ae7c4fd84f5f902880fa72079e7bb528c17a772bbcf` |
| `SHA256SUMS` file | `7fbede14f21a229259711f4fac570890649d68e8219332bce2f85cf747846f51` |
| Independent check JSON | `f10d11da4e028bde4a7b78da25e62570d80bbd1100de3e84b072b52cadf8875f` |
| Source state | `0e226030ed6d1efd17fa101a502153b9c97338ffa2b2353b8a6417fc5d3f7c36` |
| Live config bytes | `38515b79b9c1a14bc34d6f5ba9222377193f53ec0243c215926913e9e9a5ef37` |
| Labelled dataset bytes | `384457a01718b9a9121d0cf5af2289200ccd881ff764f2c5e2aef49e86ee1ba2` |
| Fixed-request matrix | `7b00317565a823a1af6b097d683273734502dd00a1129fedf715271381c2c961` |
| Deterministic schedule | `44750294110554d58af9c8a96cc8b99731d07af022aab747d39551c5a4561d51` |
| Combined artifact manifest | `33217b56eaaedb87bad9d44da90d9cd797a23197cedafeb7aa2e16dec0e8a846` |

| Stage | Model config hash(es) | Available-weight manifest | Backend executable |
| --- | --- | --- | --- |
| Qwen3.8 language | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` | `3ebc83ee545cc54cb2558d805802d4a4e9da42c316d1b248846debee545b99c9` | `d12fa5476ad11ea581517b67bf140cb7b180de3dc51d4ecaf97ad4daf392d459` |
| Qwen-Image 2.1 | five values pinned in the controlled config | `f9262ab51a1781039bbb4d750fddf29caaee25ce8880a3d4e96201a1ac694f93` | `3214a2a9807c6a4808d1a3570e63a5b0a37b35a4c92e3cf65d013ee72554c039` |
| Wan2.2 video | five values pinned in the controlled config | `9375ac1b2e4da911ddf669a30e2c4126771b785d6bbd6ca30e58acc4fca296b1` | `5cbd097e43a48b33143bfd8cb9f90ad25fa39d12a88a2acaa28010c0b256e100` |
| Qwen3-ASR 1.7B entry | `2e74a751548b8ad7d7526d29365ad8144c345d8b412b1152d25dc6698452712f` | `4b421cb776df4ae24b72ef9ac3fe08250b4eefc0d79eb0bcbece553ff38c1590` | `37fe97358f119ea895bdda2ef53424eaff7f59f6e24a41df10f94d03561452ed` |
| Gateway | N/A | N/A | `963a0a3dbdbad7b6e21e84807c2a6cd67b462b5682fc793245656c0f4bb7ee9a` |

The full evidence is retained on the isolated SM90 worker under
`/root/umm-controlled-routing-4ddf56a-live/evidence/narrative-routing-controlled-final/`.
The directory name is a workspace label; the report's actual source is the
clean `9dab363` above. After only the five recorded isolated PIDs were stopped,
GPUs 0 and 1 each returned to 3 MiB with no compute processes. The shared
container reaper left those five entries defunct/zombie; they are not running
and hold no GPU memory. No shared container or resident service was signalled
to clear them. Request-scoped allocator telemetry remains `NOT_COVERED`; the
report's device values are whole-device samples including persistent weights.

## Measured SM90 narrative-routing result — 2026-11-26

The first accepted real-model run used commit
`7e035402dce4700c72c3b44c26b5409421e86ec4` with a clean source tree and
completed all eight requests. An earlier attempt was discarded: it inadvertently
sent option objects that enabled every visual modality, then stalled in an audio
leg. It is not included in any result below. The accepted configuration corrected
selected-option activation and kept audio absent from every arm.

The isolated stack used only idle SM90 GPUs 0, 1, and 7: Qwen3.8-27B and the
required ASR entry on GPU 0, native Wan2.2 video on GPU 1, and native Qwen-Image
2.1 on GPU 7. The TTS worker was loaded on GPU 1 but received no accepted-run
request. The resident service on GPUs 2–6 and port 43301 was not used, restarted,
or contacted. FLUX.2 was not the image backend because this container's PyTorch
could not initialize CUDA; the recorded image backend is `2.1`, and the recorded
video backend is `native-cuda-wan`.

Fixed controls are seed 42, temperature 0, top-p 1, at most 256 generated tokens,
two visual slots, 512x512 four-step PNG images, and 480x320 13-frame 16 fps
four-step MP4 videos with guidance 5.0 and seed 42. The two prompts have
human-declared labels: lantern expects `image,video`; conservation expects
`image`. This is a tiny routing diagnostic, not a generalized quality benchmark.

### Routing and latency measurements

F1 is micro-averaged over each policy's two cases. Redundancy is redundant slots
divided by predicted visual slots. TTFT is the monotonic time to the first
non-empty VLM delta. Median and nearest-rank p95 therefore use only two samples
per policy; these numbers are `UNCONTROLLED_LIVE_MEASUREMENT`, not a Phase 5
performance baseline or a cold-start comparison.

| Policy | Completed | Micro F1 | Redundant-slot rate | TTFT median ms | TTFT p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `text_only` | 2/2 | 0.000 | 0.000 | 306.801 | 309.468 |
| `forced_image` | 2/2 | 0.800 | 0.000 | 362.358 | 365.067 |
| `forced_video` | 2/2 | 0.400 | 0.500 | 363.121 | 365.969 |
| `auto_tag` | 2/2 | 1.000 | 0.000 | 375.069 | 377.946 |

The auto-tag policy emitted image+video for lantern and image only for
conservation. Forced image missed the lantern video label. Forced video missed
the lantern image label and produced one redundant conservation video. TTFA is
`NOT_APPLICABLE` for every accepted-run arm. Learned routing and optimizer runs
remain `NOT_RUN`; perceptual quality and quality improvement remain unmeasured.

### Artifact and provenance evidence

`sha256sum -c SHA256SUMS` passed for every report, trajectory, and artifact, and
the terminal isolated `/health` reported `in_flight=0`. After only the recorded
isolated processes were stopped, NVIDIA reported 3 MiB on each of GPUs 0, 1, and
7 with no compute process on those GPUs. The seven visual outputs were:

| Policy / prompt | Artifact | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| forced image / lantern | PNG | 98,594 | `ea3534e9e2ef358a2973123b14f9f143a06a4eb8fbe0f28a058a2e9733e9eee9` |
| forced image / conservation | PNG | 269,919 | `5f753197ca1cbdec6cb4a68fe7a3d81d8e269a85d9a1c65adf360b8f9966d6b9` |
| forced video / lantern | MP4 | 65,966 | `303a58a36c58b5656d70f7fc2f235cdc4f4b684995f9a9eaab062ac573d7a9a4` |
| forced video / conservation | MP4 | 66,170 | `c66494fdcc2264ec3c5ca1387d4fc71688e8f7b7b400f02d1d5836bc8c05a032` |
| auto-tag / lantern | PNG | 85,337 | `9faa7e74b6f8d942c45c2673a877e429794623da532c28672d8919e946b29a1e` |
| auto-tag / lantern | MP4 | 36,883 | `f9e3d15e627c32b3f6ea5514dae3c7c7178f7ad1652de7f68cc53a84031da54d` |
| auto-tag / conservation | PNG | 221,237 | `f9f22d49b61932de5b1eba00ac482cec7a3ddb9d162e0b170f74d0474752cc89` |

The evidence directory is
`/root/umm-stage-c-28094a7-live/evidence/narrative-routing-live-final/` on the
SM90 worker. Its directory name is only an isolated-workspace label; the report's
actual source commit is the clean `7e03540` above.

| Evidence item | SHA-256 |
| --- | --- |
| Combined `report.json` | `96436a42ec2c0fe1d396f5783895f69205ef986a6da0c9d20de374f2865f76eb` |
| `SHA256SUMS` file | `765b121dc630988f5ee24784285ae0f8b61980ef95cc2938b0874206ff61e663` |
| Combined artifact manifest | `36882fab4c3943b84fd0e3e5d64723aa0ebfa449478f4649dea26b1d47ebb75e` |
| Source state | `e33d454299682c44e5b127816e94deffe75668571f8bcde019cf6c9d7dd9191e` |
| Live config bytes | `a9d22a4b0c4d91ec563d505a4b20c9c1a4bbe25d964cb7c030185d91496c1927` |
| Fixed-request matrix | `7b00317565a823a1af6b097d683273734502dd00a1129fedf715271381c2c961` |

Qwen's local `config.json` SHA-256 is
`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`.
Its 18-shard relative-filename `sha256sum` manifest SHA-256 is
`3ebc83ee545cc54cb2558d805802d4a4e9da42c316d1b248846debee545b99c9`.
Other worker weight manifests are `NOT_COVERED`, so this run does not claim full
Phase 3 reproducibility or model-quality acceptance.

## Measured SM90 visual-form result — 2026-11-26

The accepted real-model visual-form run used clean commit
`7dc2586800f82b812c551e1b4129995639492ab9`, public chat plus
`POST /v1/visual/render`, native Qwen-Image 2.1, and native C++/CUDA Wan2.2.
The isolated loopback endpoint was port 43415. Only explicitly idle GPUs 0, 1,
and 7 were assigned: ASR+VLM on 0, Wan2.2 on 1, and image on 7. No audio/TTS leg
or worker was used. The resident service and its devices were not contacted,
restarted, or assigned work.

All six first-stage cases completed, with two completed terminal image legs in
`image_only`, two completed video legs in `video_only`, and two completed image
legs plus two completed public I2V renders in `image_then_i2v_animate`. There
were no failed or cancelled cases, legs, or animations. `/health` ended with
`in_flight=0`.

| Arm | TTFT median ms | First-stage terminal median ms | Slot-to-artifact median ms | I2V terminal median ms | Peak whole-device MiB, GPU 0 / 1 / 7 |
| --- | ---: | ---: | ---: | ---: | --- |
| `image_only` | 384.510 | 15,707.466 | 11,790.863 | N/A | 61,582 / 20,859 / 28,917 |
| `video_only` | 365.242 | 9,136.093 | 5,532.940 | N/A | 61,582 / 26,533 / 28,337 |
| `image_then_i2v_animate` | 362.297 | 12,761.678 | 8,866.072 | 5,920.607 | 61,582 / 26,533 / 28,917 |

These are two-sample `UNCONTROLLED_LIVE_MEASUREMENT` values, not a Phase 5
performance baseline or evidence of a latency improvement. First-stage terminal
latency excludes the explicitly separate I2V render. GPU values are 50 ms
whole-device `nvidia-smi` samples including persistent model memory, not
request-only allocator peaks. Per-arm baseline/final values and sample counts
are in each arm's `report.json`.

Every PNG passed chunk CRC, zlib, scanline, RGB8, and 512x512 checks. Every MP4
passed full ffmpeg decode and ffprobe reported H.264, 480x320, 16 fps, and 13
frames. Independent `sha256sum -c SHA256SUMS` passed for all reports,
trajectories, and artifacts. The eight artifact occurrences (six unique blobs,
because the two I2V source PNGs repeat byte-identically) were:

| Arm / prompt | Artifact | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| image-only / lantern | PNG | 98,594 | `ea3534e9e2ef358a2973123b14f9f143a06a4eb8fbe0f28a058a2e9733e9eee9` |
| image-only / conservation | PNG | 269,919 | `5f753197ca1cbdec6cb4a68fe7a3d81d8e269a85d9a1c65adf360b8f9966d6b9` |
| video-only / lantern | MP4, 13 frames | 65,966 | `303a58a36c58b5656d70f7fc2f235cdc4f4b684995f9a9eaab062ac573d7a9a4` |
| video-only / conservation | MP4, 13 frames | 66,170 | `c66494fdcc2264ec3c5ca1387d4fc71688e8f7b7b400f02d1d5836bc8c05a032` |
| I2V / lantern source | PNG | 98,594 | `ea3534e9e2ef358a2973123b14f9f143a06a4eb8fbe0f28a058a2e9733e9eee9` |
| I2V / lantern animation | MP4, 13 frames | 12,026 | `b8f0e50391eacf337ec44918e0ec1c2e5ff3e183ce1c22448c5260874f7c14c6` |
| I2V / conservation source | PNG | 269,919 | `5f753197ca1cbdec6cb4a68fe7a3d81d8e269a85d9a1c65adf360b8f9966d6b9` |
| I2V / conservation animation | MP4, 13 frames | 31,396 | `d20f5681613646d249723cfbccd2a5c6dfb95b22123df2aac355daf56f6f5cf2` |

### First-stage identity and I2V behavior

| Prompt | Shared image-only/I2V request SHA-256 | Shared source PNG SHA-256 | Request and source bytes identical |
| --- | --- | --- | --- |
| lantern | `3d62026b3913f20308230738640585db7714713bb83a7e4ccae5bce8c3f27028` | `ea3534e9e2ef358a2973123b14f9f143a06a4eb8fbe0f28a058a2e9733e9eee9` | yes |
| conservation | `843c28c8365882ac1fd4a2f5562347da477e7663f8ec23b57987224896511cf2` | `5f753197ca1cbdec6cb4a68fe7a3d81d8e269a85d9a1c65adf360b8f9966d6b9` | yes |

Each animation's `source_image_sha256` equals its source PNG's CAS hash. The
render-request SHA-256 values were
`24b9fec06904a69cefc5e2336957ebb138868df3479560953610da4ec72a959f` (lantern)
and `17b4013963213aba961d80476d0aa9ed4e0e54fa51f777fb50f12f106f5d6381`
(conservation).

| I2V prompt | Center-crop MAE | Stretch MAE | Top-left-crop MAE | First-frame PSNR dB | Behavioral gate |
| --- | ---: | ---: | ---: | ---: | --- |
| lantern | 1.083559 | 16.687817 | 40.043331 | 45.369568 | PASS |
| conservation | 1.924601 | 14.183991 | 31.584082 | 35.398458 | PASS |

Both decoded first frames were closer to the native half-pixel center-crop
reference than to either alternative. This validates the requested
first-frame/crop behavior for these two cases only. It is not a perceptual
quality label, user preference study, or generalized I2V quality claim.

### Provenance and limits

Evidence is retained on the isolated SM90 worker under
`/root/umm-visual-form-7dc2586-live/evidence/visual-form-live-7dc2586/`.
Weight manifests are in its sibling `evidence/provenance/` directory.

| Evidence item | SHA-256 |
| --- | --- |
| Combined `report.json` | `b63ca8ab9290804a43101cb054805f7c2c6a7c00f2ad6fb9b9c9e369d048cb66` |
| `SHA256SUMS` file | `21731a15ff13d1d3e93d38add82a9a2fa54a7c08b26dcde858c2dd1244f4f0fd` |
| Source state | `8ba91dcf56da64aa0ee43098587033347afb0dfb96c81245c3211a2463035cf7` |
| Live config bytes | `4bace54699829f4ac45227273dbdf5b59519a9b0c371d7ee580cd96e86f8019d` |
| Prompt dataset bytes | `9279dda6c793d7e553d977f515415ad2c9e46fb639d49958935de7697850ba44` |
| Combined artifact manifest | `9f72852bf842604c1fcaf89d1eaac813a505dc2cc1bc3ade2e13f5e5a384a1bf` |
| Qwen3.8 18-file weight manifest | `3ebc83ee545cc54cb2558d805802d4a4e9da42c316d1b248846debee545b99c9` |
| Qwen-Image 2.1 seven-file weight manifest | `f9262ab51a1781039bbb4d750fddf29caaee25ce8880a3d4e96201a1ac694f93` |
| Wan2.2 nine-file weight manifest | `9375ac1b2e4da911ddf669a30e2c4126771b785d6bbd6ca30e58acc4fca296b1` |

The tracked live manifest also carries the actual model/config-file hashes.
Local snapshots have no verified upstream revision ID. Two fixed prompts,
uncontrolled latency, polling-based whole-device memory, and no perceptual or
preference labels limit this result to reproducible protocol/artifact and
algorithmic behavior acceptance. Image-text relevance, generalized visual
quality, learned routing, and optimizer improvement remain uncovered or unrun.

## Controlled SM90 visual-form result — 2026-11-26

The remaining visual-form gate ran against newest-main base
`08fb5e9e013879ecc75d2194e61bd21534a7fc55`, focused runner commit
`d9f645cab7b48c31c82c6d4ba38b2bdfed89cfdc`, and clean measurement source
`23358c74ba186cd86caab9671a45200ddbd12d95`. The strict command included
`--require-controlled` and exited 0. The combined verdict is
**CONTROLLED_LIVE / COMPLETED** with no eligibility refusal reasons.

Only explicitly idle physical GPU 0 (NVIDIA H20, 97,871 MiB) was assigned. The
isolated stack used an ephemeral loopback endpoint, fresh IPC paths, Qwen3.8 and
native Wan2.2 resident on GPU 0, and native Qwen-Image 2.1 with its existing
transient per-component mode (`UMM_IMAGE_RESIDENT_CACHE=0`). Wan used
`UMM_VAE_NO_PERSIST=1` so its VAE did not remain resident between calls. A
no-audio echo entry served only the existing `/ready` topology requirement;
every accepted request went directly to the real VLM and no ASR/TTS model or
audio leg was used. GPU 1, GPU 7, GPUs 2–6, and the resident service on port
43301 were not assigned, contacted, restarted, or used.

Excluded initialization calls initialized text decode, transient image, T2V,
and I2V before the baseline. A second post-I2V image proved steady-state
transient operation and byte-identical PNG output. An earlier rejected
initialization request and a discarded persistent-VAE topology attempt were
not part of warmup, measurement, or the accepted evidence directory.

The deterministic schedule hash is
`7ac02847912006312ac4b873dbdc020d9a80512b11f035a4659d7f5578a55311`.
Its 12 arm steps comprise three warmup steps followed by nine measured steps
in seeded, repetition-major interleaved order. Every step executed both fixed
human-labelled prompts. This produced 6 excluded warmup chat cases plus 2
warmup I2V renders, and 18 measured chat cases plus 6 measured I2V renders.
Every arm/prompt pair completed repetitions 1, 2, and 3 with a byte-identical
request. All 12 cleanup checks reported `in_flight=0` and residual memory no
more than 128 MiB above the initialized baseline.

### Controlled latency and resource measurements

The following pooled cells have six measured samples per arm; every per-prompt
summary in the JSON has 3/3 samples. p95 is nearest-rank (therefore the maximum
at n=3 or n=6), and IQR/MAD are milliseconds of dispersion. First-stage
terminal latency excludes the explicitly separate I2V render. TTFA is
`NOT_APPLICABLE` because no arm has audio.

| Arm | Measured / warmup cases | TTFT median / p95 ms | TTFT IQR / MAD ms | First-stage terminal median / p95 ms | Terminal IQR / MAD ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `image_only` | 6/6 + 2/2 | 313.626 / 314.834 | 2.413 / 1.202 | 18,094.266 / 18,551.686 | 765.125 / 382.563 |
| `video_only` | 6/6 + 2/2 | 314.346 / 315.224 | 2.092 / 0.844 | 6,356.763 / 6,400.706 | 94.521 / 40.555 |
| `image_then_i2v_animate` | 6/6 + 2/2 | 313.458 / 314.833 | 2.140 / 1.070 | 17,871.221 / 18,562.096 | 639.257 / 319.629 |

| Arm | Slot-to-artifact median / p95 ms | Slot IQR / MAD ms | I2V render median / p95 ms | I2V IQR / MAD ms |
| --- | ---: | ---: | ---: | ---: |
| `image_only` | 15,780.881 / 16,068.645 | 677.478 / 198.313 | N/A | N/A |
| `video_only` | 4,221.170 / 4,302.983 | 55.861 / 26.191 | N/A | N/A |
| `image_then_i2v_animate` | 15,655.102 / 15,900.354 | 641.947 / 206.011 | 3,426.626 / 3,473.888 | 60.583 / 28.237 |

The image-bearing terminal times include transient component loading in this
single-device topology. These are controlled observations for this exact
binary, two-prompt diagnostic, and low-memory configuration, not a Phase 5
baseline or a generalized latency-improvement claim.

| Arm | GPU 0 baseline MiB | Observed peak MiB | Final MiB | Actual 50 ms poll count |
| --- | ---: | ---: | ---: | ---: |
| `image_only` | 78,064 | 92,558 | 78,106 | 1,566 |
| `video_only` | 78,064 | 85,038 | 78,106 | 560 |
| `image_then_i2v_animate` | 78,064 | 92,558 | 78,106 | 1,975 |

These are whole-device `nvidia-smi` samples including persistent weights and
other interleaved arms, not request-scoped allocator peaks. Request-scoped
telemetry remains `NOT_COVERED`. The final loaded residual was 42 MiB.

### Controlled artifacts, source linkage, and first-frame behavior

All 18 first-stage terminal states and all 18 terminal visual legs were
`completed`; all six measured animations returned HTTP 200 and `completed`.
There were no leg errors, failed measured cases, or cancelled cases. Every PNG
passed CRC, zlib, scanline, RGB8, and 512x512 checks. Every MP4 passed full
ffmpeg/ffprobe decode as H.264, 480x320, 16 fps, and 13 frames. The six unique
measured visual outputs were:

| Arm / prompt | Artifact | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| image-only + I2V source / lantern | PNG | 101,319 | `502e44bf55c77f9b8ae296344e58ac3231a94ef1b73d9b57de30ae2307303c49` |
| image-only + I2V source / conservation | PNG | 133,330 | `121b481be39de49e463e424dadf0c4e258ceadf1acb1e73c649407ecd68b5ae7` |
| video-only / lantern | MP4, 13 frames | 42,019 | `00910cba59c7f76797b6792d1db9b59a71cf20cf257370ea3d9bc9c892f5b13e` |
| video-only / conservation | MP4, 13 frames | 89,475 | `6376650b2680e6281cdfaae82270d3cf1afd767ec5ec0d4806ee8fdc752994b0` |
| I2V animation / lantern | MP4, 13 frames | 13,755 | `1d46c84fef2efe2f5b1c1086a3a68e1ecc74c70a46615806fe4a820776cfbbd4` |
| I2V animation / conservation | MP4, 13 frames | 14,018 | `3705a00b1b2be6f2a3e5710c6cbcff1d51626112175a442b53df92bb7019acb8` |

For each prompt and each of three repetitions, image-only and I2V used the same
first-stage request and exact source PNG. The shared request SHA-256 values are
`3d62026b3913f20308230738640585db7714713bb83a7e4ccae5bce8c3f27028`
(lantern) and
`843c28c8365882ac1fd4a2f5562347da477e7663f8ec23b57987224896511cf2`
(conservation). Every animation's `source_image_sha256` matches both its CAS
source and the corresponding image-only source. Render-request hashes are
`19325dd979c683c4d577a2f8a27112b29721886fd0ebaac9d75e16334371a96d`
and `bf88731e0329affe9346b2bf71ceb04504618a12a5bfd7fdb6c4df4dbdfdb16f`,
respectively.

| I2V prompt | Repetitions | Center-crop MAE | Stretch MAE | Top-left-crop MAE | First-frame PSNR dB | Behavioral gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| lantern | 3/3 | 1.023207 | 11.360968 | 22.067606 | 46.076711 | PASS |
| conservation | 3/3 | 1.253698 | 3.884264 | 7.431113 | 44.442570 | PASS |

These are algorithmic center-crop/anchor behavior proxies only. Perceptual
quality, image-text relevance, user preference, and generalized improvement
remain unclaimed and `NOT_COVERED`.

### Controlled provenance and cleanup evidence

All 34 available Safetensors files were rehashed successfully: Qwen3.8 18/18,
Qwen-Image 2.1 7/7, and Wan2.2 9/9. The manifest hashes match the tracked
configuration. The four frozen executable SHA-256 values are:

| Executable | SHA-256 |
| --- | --- |
| Gateway | `e7286766f6e73532425a691ade332ccae5e2a993457075eb1032073fec21f739` |
| Qwen3.8 | `d08eddcff9f3ac5aa5c2dcecb80b97e1b3d155805add4dd983aca2ce66b185e6` |
| Qwen-Image 2.1 | `4494b2d58b1d280b273009779f098cbdffa0c0acbc839eebec9b0dde5c2ec735` |
| Wan2.2 | `6e13e9cda2771c230eef19fd8fe50913b7101024bb15fdb5ce5787eba359e4f8` |

| Evidence item | SHA-256 |
| --- | --- |
| Combined `report.json` | `8eef3c02164979d235d83c4a3f4daba21a29bad7aab20d77f2385e15e01ba957` |
| `SHA256SUMS` file | `fe534a74713860a7d6c4a7aaa2b79b7f49a3e6ce593ef1e697907d8cbf661826` |
| Independent check JSON | `6749b951043d11ffd1a3e1d00c7c4e2c3efa4d06eef54847deffe4dfa7ca5685` |
| Weight verification JSON | `a9d872342b2c257f1940b95d70950705b017126423651b817891c01531642e2b` |
| Source state | `c9507778cf58d45cb967d3eda5359daa6aad8e21dd5c58d94c5dc0f55edac287` |
| Live config bytes | `941781a33a4489ca3ffd39de13d4d3abe5593f5c089c25d8266896f55fc488d6` |
| Labelled dataset bytes | `384457a01718b9a9121d0cf5af2289200ccd881ff764f2c5e2aef49e86ee1ba2` |
| Fixed-request matrix | `7b00317565a823a1af6b097d683273734502dd00a1129fedf715271381c2c961` |
| Deterministic schedule | `7ac02847912006312ac4b873dbdc020d9a80512b11f035a4659d7f5578a55311` |
| Combined artifact manifest | `068f929c62986e01c56189edc986e61776ce57b634727fd5992495d499c00398` |

An independent checker passed 78 assertions over all 18 measured cases, six
warmup cases, six measured I2V renders, 24 measured artifact occurrences,
source links, per-repetition equivalence, schedule/eligibility/resource rows,
and cleanup. `sha256sum -c SHA256SUMS` separately passed all 26 covered files.
Evidence remains on the isolated SM90 worker under
`/root/umm-controlled-visual-26-final-live/evidence/visual-form-controlled-final/`.

After only the five recorded isolated container PIDs were stopped, GPU 0
returned to 3 MiB with no compute processes. Its six isolated IPC socket files
were removed. The shared container reaper left those entries as zombies; they
are not running and hold no GPU memory. No shared reaper, other GPU, or resident
service was signalled.

## Verification

```text
python3 -I tests/ablation_runner_test.py
python3 -I tests/controlled_measurement_test.py
python3 -I tests/rl_architecture_test.py
python3 -I tests/rl_phase2_test.py
python3 -I tests/narrative_live_e2e.py --self-check
python3 -I tests/multimodal_benchmark_test.py
python3 -I tools/check_feature_map.py
```

The ablation tests cover full-tree replay reproducibility, request invariance,
label gating and reward arithmetic, hash/policy/response/render tamper rejection,
live out-of-band timing, live-versus-replay trajectory byte identity, I2V
source-image binding and animation completion, real PNG/H.264 decoder behavior,
center-crop comparison, deterministic resource accounting, rejection of invalid
contract media by the real decode gate, and serving isolation. The controlled
measurement tests add contract/tamper validation, seeded schedule
reproducibility and schedule-hash tamper detection, single-variable invariance
across arms and repetitions, latency median/p95/IQR/MAD aggregation with sample
counts, incomplete-repetition and failed-arm refusal, terminal/leg validity,
exclusive-device precondition failure and fail-fast abort, stuck-cleanup
refusal, incomplete language/visual/auxiliary provenance refusal,
per-repetition visual-form request/source equivalence with explicit
`CONTROLLED_LIVE` refusal, and true polled resource peaks/sample counts.
Loopback canned responses remain
contract-only; decoder tests use independently generated tiny media and do not
access a GPU, remote host, real model, optimizer, or resident service.
Controlled loopback runs inject device/health samplers and are not real
controlled performance evidence; the separate accepted real gate is documented
above.
