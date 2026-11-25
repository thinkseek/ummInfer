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
completion only. Image-text relevance, first-frame perceptual consistency, and
user blind preference remain `NOT_COVERED`; no numerical visual-quality claim is
made by the harness.

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
```

Live mode uses the existing out-of-process schema-v2 recorder. An optional
normalized-event observer measures monotonic latency outside the exported
record. No C++ serving file imports the runner, metrics, reward, trainer, or
optimizer. Live reports are `live-uncontrolled`; a real performance comparison
still needs an exclusive measurement window and a real pinned model. For real
model provenance, set a non-null `model.weights_sha256`; the default contract
model has no weights and reports weights coverage as `NOT_COVERED`.
`model.config_sha256` hashes the runner's `{id,revision}` identity descriptor,
not an external model card or weight file.

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
```

Each trajectory is validated by the existing independent RL worker after CAS
reassembly and reward annotation. Media bytes stay out of JSONL. `SHA256SUMS`
covers every report, trajectory, and artifact. Two replay runs from the same
commit and inputs produce byte-identical relative file trees; the regression
test compares the complete trees, not just summary values.

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

The checked-in replay reports `CONTRACT_OBSERVATIONS_ONLY`, learned routing
`NOT_RUN`, optimizer `NOT_RUN`, quality improvement `NOT_COVERED`, and real model
quality `NOT_RUN`. No latency, F1, redundancy, quality, learned-routing, or
optimizer improvement claim may be made from these canned fixtures.

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

## Verification

```text
python3 -I tests/ablation_runner_test.py
python3 -I tests/rl_architecture_test.py
python3 -I tests/rl_phase2_test.py
python3 -I tests/narrative_live_e2e.py --self-check
python3 -I tests/multimodal_benchmark_test.py
python3 -I tools/check_feature_map.py
```

The ablation tests cover full-tree replay reproducibility, request invariance,
label gating and reward arithmetic, hash/policy/response/render tamper rejection,
live out-of-band timing, live-versus-replay trajectory byte identity, I2V
source-image binding and animation completion, and serving isolation. They use
loopback canned responses only and do not access a GPU, remote host, real model,
optimizer, or resident service.
