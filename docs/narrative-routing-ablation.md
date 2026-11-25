# Stage C narrative-routing ablation

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

The config pins the dataset and replay byte SHA-256. Every replay case also pins
its full request hash, fixed-request hash, and raw response hash. A config,
prompt, sampling, visual-setting, response, or policy edit that is not matched by
the corresponding hash fails closed.

## Single-variable rule

`build_request()` constructs every policy request from the same model, messages,
sampling, audio, image, and video objects. Only `modalities` and `narrative` are
overlaid. `fixed_request_sha256` hashes the complete request except those two
fields. The report contains one such hash per prompt and a
`fixed_matrix_sha256` over all prompts; these values must agree across policies.

The checked-in fixed values include seed 42, temperature 0, top-p 1, 256 maximum
tokens, the same streaming-audio settings, two visual slots, and identical
image/video sizes, steps, guidance, and seeds. Options remain present in every
request even when their modality is disabled by the routing overlay.

## Run

CPU-only deterministic replay:

```text
python3 -I python/umminfer_ablation_run.py \
  --config benchmarks/ablations/narrative-routing.json \
  --replay-only --out /tmp/umminfer-narrative-routing-ablation
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
label gating and reward arithmetic, hash/policy/response tamper rejection, live
out-of-band timing, live-versus-replay trajectory byte identity, and serving
isolation. They use loopback canned responses only and do not access a GPU,
remote host, real model, optimizer, or resident service.
