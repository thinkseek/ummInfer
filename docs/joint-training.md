# Stage C Joint SFT & Preference (DPO / GRPO) Dataset Builder

This document specifies the offline Stage C dataset builder and quality/leakage
auditor in `python/umminfer_sft_dataset.py`. It implements the data-packing
boundary between `ummInfer` trajectory exports and downstream joint
post-training described in [vision.md](vision.md), [rl.md](rl.md), and
[narrative-routing-ablation.md](narrative-routing-ablation.md).

## Architectural Boundary

- **Strict serving isolation**: `python/umminfer_sft_dataset.py` uses pure
  Python 3 stdlib only (no `torch`, `numpy`, or `transformers`) and imports no
  serving or inference runtime code. No C++ serving file under `src/` or
  `include/` imports it.
- **Read-only ingestion**: trajectory JSONL files (schema v1 and schema v2) and
  their content-addressed `artifacts/` directories are read-only inputs. Media
  bytes are never embedded in dataset JSONL outputs; only verified SHA-256
  content references are preserved.
- **No optimizer execution**: this tool builds and audits training packs only.
  Every generated manifest explicitly marks `REAL_TRAINING: NOT_RUN`,
  `optimizer: NOT_RUN`, and `model_weights_updated: false`.
- **Optional token/span sidecar in the v2 record**:
  `python/umminfer_token_export.py` is a separate stdlib-first offline utility,
  not a trainer or serving dependency. It adds exact tokenizer ids, UTF-8 byte
  spans, raw-source mappings, and `joint_full` mask weights only when explicitly
  invoked; records without `token_export` retain their previous behavior.

## Integrity Verification & Filtering

Every input line is verified before entering dataset construction:

1. **Structural and digest checks** (schema v1 and v2):
   - Supported integer `schema_version` (`1` or `2`; booleans rejected).
   - Valid `terminal_state` (`completed`, `failed`, `cancelled`), gap-free event
     `seq`, and recomputed `output_sha256`.
   - For schema v2: gap-free `server_seq`, single final terminal event matching
     `terminal_event_seq` and `terminal_legs`, recomputed `clean_text_sha256`
     matching narrative deltas, validated `visual_slots` and `leg_errors`,
     complete chunk hash chains (`chunks_sha256`), no embedded base64 media
     payloads, and valid relative CAS references bound to their SHA-256.
     When `token_export` is present, its source-record binding, tokenizer/file
     identity, ids, exact offsets, source mappings, spans, masks, and hashes are
     also validated; malformed exports are rejected as `token_export_tampered`.
2. **Content-addressed media verification**:
   - When an artifact directory is present (either via `--artifact-dir` or the
     default sibling `artifacts/` next to the JSONL file), every referenced
     input, request-parameter, and output CAS file is checked for existence,
     byte length, and SHA-256 equality. Tampered or missing artifacts are
     rejected as `unverified`.
3. **Terminal-state gating**:
   - Only verified records with `terminal_state == "completed"` are accepted.
   - Verified `cancelled` and `failed` trajectories are excluded and tracked in
     the manifest's rejection counters (`cancelled`, `failed`, `unverified`,
     `duplicate`, `unscored`).

## Pack Formats

### 1. SFT Interleaved Modality-Routing Pack (`--mode sft`)

Outputs `sft_train.jsonl` and `sft_eval.jsonl`.

- **Director-supervised target reconstruction**:
  - For schema-v2 turns with `visual_slots`, canonical beat markers
    `[[image: <prompt>]]` and `[[video: <prompt>]]` are inserted into
    `clean_text` at each slot's UTF-8 byte `text_offset`, preserving an exact
    round-trip with `NarrativeStreamParser`.
  - For text-only v2 turns (`visual_slots == []`) and schema-v1 turns, the
    target is clean prose with no visual markers.
- **Media provenance**: verified `input_media_refs`, `output_media_refs`, and
  per-slot `leg_state` / `artifact_sha256` bindings are preserved without
  inlining binary bytes.
- **Deduplication & determinism**: candidates are deduplicated by canonical
  `(prompt_sha256, target_sha256)` in deterministic tie-breaker order and
  emitted as key-sorted compact UTF-8 JSONL (`allow_nan=False`).

#### Optional exact tokenizer ids and structured target spans

After all trajectory enrichment is final, the token-export CLI can create a new
JSONL with additive v2 `token_export` blocks while leaving v1 rows untouched:

```text
python3 -I python/umminfer_token_export.py trajectories.jsonl \
  --out trajectories.tokens.jsonl \
  --tokenizer-dir /path/to/local/tokenizer \
  --tokenizer-id org/model-or-fixture-id \
  --tokenizer-revision immutable-revision-or-local-fixture-version \
  --tokenization-version transformers-and-tokenization-version
```

The four tokenizer options are mandatory together. The utility hashes every
available regular tokenizer file, uses no network or model inference, disables
special-token insertion, padding, and truncation, and rejects unknown or
non-byte-exact tokens. It exports both `clean_text` and the reconstructed
`assistant_target`. Every token has an exact half-open UTF-8 byte range and
normalized source event indices; replay-enabled records also carry exact raw SSE
frame indices. Synthesized route headers, prompt bodies, explicit-anchor
suffixes, and closing delimiters map to their source `visual_slot` frame.

The target stream's typed spans and per-token weights are bound to the same
`joint_full` contract used by `build_span_loss_mask` and the bounded real SFT
adapter. The export stores both byte and codepoint offsets: bytes provide exact
source reconstruction, while codepoint-overlap weighting preserves the existing
SFT mask when a token crosses span types. Deterministic tests compare this
export against `reconstruct_director_target`, `parse_interleaved_spans`, and the
real adapter's `_token_loss_mask`, including route/body/closing coverage and
multibyte text. The full schema, strong re-encode/decode validation command, and
hash/tamper rules are in [rl.md](rl.md#optional-token-id-and-source-span-export-format-v1).

This export does not change dataset JSONL version 1, dataset deduplication,
splitting, scores, pair identities, or GRPO packing. DPO chosen/rejected response
summaries additively retain the same export and complete source binding when
present; preference rows without exports retain their exact previous JSONL.
The existing worker and dataset integrity checks reject a malformed block whenever present. Structural validation without
a tokenizer is not a proof of tokenizer identity; the optional real-tokenizer
fixture gate performs that re-encoding and remains `NOT_RUN` when no explicit
local fixture is supplied.

For the actual Qwen fast tokenizer's NFC composition and byte fallback, the
explicit `--token-export-version 2` format stores exact normalized token bytes,
overlapping original-source units, and separate raw offsets for the unchanged
SFT mask calculation. V1 output and rejection behavior remain unchanged;
unsupported normalization/vocabulary behavior fails closed. The designated
`Qwen/Qwen3.8-27B` local-file gate **PASS** covers multilingual, emoji, combining,
spoken, visual-slot, explicit-anchor, and cancelled records, plus two fresh CLI
processes. Exact tokenizer files, hashes, commands, and boundaries are in
[token-export-real-qwen.md](token-export-real-qwen.md) and the complete v2
contract is in [rl.md](rl.md#format-v2-nfc-normalization-and-byte-fallback-provenance).
No model inference, sampled decode ids, online rollout, or optimizer result is
claimed.

### 2. Preference (DPO) & GRPO Group Pack (`--mode preference`)

Outputs `dpo_train.jsonl`, `dpo_eval.jsonl`, `grpo_train.jsonl`, and
`grpo_eval.jsonl`.

- **Export-only reward scoring**: trajectories are grouped by canonical
  `prompt_sha256` and scored strictly from the exported JSON record using
  `umminfer_rl_reward.load_reward("module:attr")`. The default reward is
  `umminfer_ablation_metrics:NarrativeRoutingReward` (`F1 - redundant_slot_rate`
  on labelled exports; unlabelled records return `None` and are counted under
  `rejected.unscored`).
- **Pairwise DPO records**: distinct-target rollout pairs `(chosen, rejected)`
  within each prompt group are emitted deterministically whenever
  `chosen.score > rejected.score` and
  `margin = chosen.score - rejected.score >= --min-margin`.
- **GRPO group records**: every prompt group with at least `--min-group-size`
  (default `2`) scored rollouts computes group-relative normalized advantages:
  - `advantage_i = (score_i - mean) / std` when `std > 0`,
  - `advantage_i = 0.0` when `std == 0` (zero-variance group, never NaN/Inf).

Using `--mode both` emits all six JSONL files (`sft_*`, `dpo_*`, `grpo_*`) in
one audited pass with identical prompt-level split assignment.

## Deterministic Split, Leakage Auditor & Provenance Manifest

- **Prompt-level split**: each prompt is routed to `train` or `eval` using
  `sha256(str(seed) + prompt_sha256)`. All SFT examples, DPO pairs, and GRPO
  groups sharing a `prompt_sha256` always land in the same split.
- **Zero-leakage check**: both build time and `--audit-only` independently
  verify that `train_prompts ∩ eval_prompts == ∅` (`leakage_count == 0`).
- **Provenance manifest**: writes `manifest.json` and `manifest.json.sha256`
  containing input file SHA-256 digests, artifact verification mode and counts,
  filter rejection counters, output JSONL digests and record counts, split
  leakage audit results, and `REAL_TRAINING: NOT_RUN`. Two runs over the same
  inputs and flags produce byte-identical output trees.

## CLI Usage

Build an SFT pack from schema-v1/v2 trajectories:

```text
python3 -I python/umminfer_sft_dataset.py \
  --input path/to/trajectory.jsonl \
  --out path/to/sft-pack \
  --mode sft \
  --seed umminfer-stage-c-v1 \
  --eval-ratio 0.2
```

Build DPO + GRPO packs with the default narrative-routing reward (or a custom
`module:attr` reward):

```text
python3 -I python/umminfer_sft_dataset.py \
  --input path/to/policy_a.jsonl path/to/policy_b.jsonl \
  --out path/to/pref-pack \
  --mode preference \
  --reward umminfer_ablation_metrics:NarrativeRoutingReward \
  --min-margin 0.1 \
  --min-group-size 2 \
  --seed umminfer-stage-c-v1 \
  --eval-ratio 0.2
```

Independently audit an existing dataset pack directory (`manifest.json.sha256`,
output JSONL hashes, split labels, and zero prompt leakage):

```text
python3 -I python/umminfer_sft_dataset.py \
  --audit-only \
  --out path/to/pref-pack
```

## Stage C Live-to-Training E2E Pipeline Driver (`tools/umminfer_stage_c_pipeline.py`)

`tools/umminfer_stage_c_pipeline.py` bridges exported `umminfer_rl_trajectory`
schema-v1/v2 JSONL trajectories directly through `python/umminfer_sft_dataset.py`
and `python/umminfer_joint_trainer.py` in a single content-addressed, stdlib-only
pass (100% decoupled from C++ serving code under `src/`):

1. **Strict trajectory, raw-replay & CAS validation**: fails closed on malformed
   JSON, unsupported `schema_version`, tampered text/event/source-frame/tick
   digests or sequences, and missing/tampered content-addressed media or `.sse`
   sidecars. Replay-enabled records report `replay_records` and
   `verified_replay_refs` in the input audit.
2. **Dataset packing & leakage audit**: builds `sft/`, `preference/` (DPO + GRPO
   with pluggable `--reward`, defaulting to
   `umminfer_rl_reward:CompositeUmmReward`), and cross-verified `joint/` packs,
   enforcing `leakage_count == 0`.
3. **Trainer / ablation-matrix execution**: invokes `umminfer_joint_trainer.py`
   across a single `--config` or the full `--ablation-matrix`
   (`configs/training/ablation_matrix.json`) in `dry_run_micro_step` (default),
   `plan`, or `gpu_launch_script` mode, binding the freshly built pack manifests.
4. **Content-addressed report**: writes `stage_c_e2e_report.json` and
   `stage_c_e2e_report.json.sha256` with explicit `REAL_TRAINING: "NOT_RUN"`.

Offline mode (default workflow over exported JSONL files or directories):

```text
python3 -I tools/umminfer_stage_c_pipeline.py \
  --from-trajectories path/to/trajectories_a.jsonl path/to/export_dir \
  --out path/to/stage-c-e2e \
  --reward umminfer_rl_reward:CompositeUmmReward \
  --ablation-matrix configs/training/ablation_matrix.json \
  --execution-mode dry_run_micro_step \
  --dry-run-steps 2
```

Opt-in live capture bridge (`--capture-live <base_url>`): sends a deterministic
multimodal prompt suite (pure-text with `allow_text_only: true`, visual-scene
image, visual-scene video/auto, and mixed narrative beats, 2 contrasting rollouts
per prompt) to `POST /v1/chat/completions` (`stream: true`), captures exact raw
SSE chunks/frames plus observed monotonic arrival ticks into versioned `.sse`
sidecars, parses verified schema-v2 JSONL records with local CAS media artifacts,
and executes the identical SFT -> Preference -> Leakage Audit -> Joint Trainer
pipeline end-to-end (`source_mode: "live_capture"`, `REAL_TRAINING: "NOT_RUN"`).
TTFT/TTFA use the captured arrival ticks, not the deferred normalization time.
Offline replay is `python3 -I python/umminfer_sse_replay.py INPUT --out OUTPUT`;
the complete schema and clock contract are in [rl.md](rl.md#raw-sse-byte-replay-sidecar-format-v1).

```text
python3 -I tools/umminfer_stage_c_pipeline.py \
  --capture-live http://127.0.0.1:43301 \
  --model qwen3-native \
  --out path/to/stage-c-live-e2e \
  --ablation-matrix configs/training/ablation_matrix.json \
  --execution-mode dry_run_micro_step
```

## Verification & Claim Boundary

Run the offline CPU-only contract tests:

```text
python3 -m unittest tests/stage_c_pipeline_test.py tests/sft_dataset_test.py tests/joint_trainer_test.py
python3 -I tests/sse_replay_test.py
python3 -I tests/token_export_test.py
python3 -I tests/narrative_slot_reference_live.py --self-check
python3 -I tests/joint_sft_live_test.py
python3 -I tests/joint_dpo_live_test.py
python3 -I tests/joint_grpo_live_test.py
python3 -I tests/online_rollout_test.py
```

| Capability | Status | Notes |
| --- | --- | --- |
| Offline v1/v2 trajectory integrity & CAS audit | **COVERED** | Deterministic CPU-only unit tests |
| Raw-SSE live-versus-replay byte identity | **COVERED (loopback fixture)** | Exact raw/chunk/frame/[DONE], logical ticks, normalized JSONL and source hashes; no model claim |
| Isolated real-service raw capture + fresh-process replay | **PASS (SM90 GPU 0)** | Real text/visual-contract/cancelled streams, observed `monotonic_ns`, byte-identical JSONL/CAS and tamper rejection; visual media are contract fixtures. See [sse-replay-live.md](sse-replay-live.md) |
| SFT interleaved beat-marker reconstruction & dedup | **COVERED** | Byte-identical across repeated runs |
| Optional token-id/UTF-8/source-span export & SFT mask equivalence | **COVERED (fixture v1/v2); PASS (real local Qwen v2)** | Additive blocks, fail-closed structural validation, exact normalized bytes/source units, deterministic independent decode and two fresh CLI processes; default run without a supplied tokenizer still reports `NOT_RUN` |
| Export-only DPO pair & GRPO group-advantage packing | **COVERED** | Pluggable `load_reward("module:attr")`; additive DPO-side export/source retention |
| Exported-token DPO pair data/preflight | **PASS (fixture v1/v2 + real local Qwen2.5 v2 CPU)** | Two real preference pairs, all-or-nothing validation, exact policy/reference packing, run identity, five fail-closed real negative cases, and byte-identical fresh-process verification; real optimizer remains NOT_RUN. See [exported-token-dpo-preflight](exported-token-dpo-preflight.md) |
| Zero train/eval prompt leakage & manifest sidecar | **COVERED** | Verified at build time and `--audit-only` |
| End-to-end trajectory -> pack -> ablation-matrix dry-run driver | **COVERED** | `tools/umminfer_stage_c_pipeline.py` + `tests/stage_c_pipeline_test.py` |
| Real joint-SFT optimizer gate | **PASS (SM90, legacy + token_export)** | Two exact-equivalent AdamW steps per path, changed finite parameters, fresh-process export resume, and pre-output invalid-input rejection; see [exported-token-sft-live](exported-token-sft-live.md) |
| Real joint-DPO optimizer gate | **PASS (SM90, historical legacy path)** | Two real sigmoid-DPO AdamW steps, policy/reference log-probabilities, independent chosen/rejected masks, exact resume equivalence, and pre-import malformed/crossed-pair rejection; export-driven real optimization remains NOT_RUN |
| Real joint-GRPO optimizer gate | **PASS (SM90)** | Two complete export-bound three-rollout groups, clipped policy ratios + KL, finite AdamW updates, exact fresh-process resume, and six pre-import rejection gates |
| Online GRPO orchestration and validation | **COVERED (fake endpoint)** | Deterministic requests, exact response attestation, raw SSE, cancellation, bounded retry, complete groups, and fresh-process byte replay; no real-model claim |
| Real online generation and online optimizer execution | **NOT_RUN** | Requires the isolated live gate below; the accepted frozen-offline GRPO result is unchanged |
| Endpoint behavior log-probabilities and masks | **NOT_COVERED by the current public endpoint** | Null with `NOT_COVERED` when absent; complete endpoint-supplied arrays are structurally validated when exposed |

## Online GRPO rollout generation and validation contract

`python/umminfer_online_rollout.py` is a trainer-side HTTP orchestrator. It does
not start a server, load a model, import an inference stack, or import Torch,
NumPy, Transformers, or optimizer code. It reuses the public schema-v2 recorder,
raw-SSE replay, export-only reward interface, CAS verification, and deterministic
Stage C GRPO packer. No serving source imports this module or the dataset
builder. The accepted frozen-offline adapter explicitly rejects rows and source
trajectories carrying online provenance, including ordinary packs rebuilt from
online JSONL, so the two evidence paths cannot be silently interchanged.

The versioned plan (`umminfer-online-grpo-plan-v1`) fixes the batch id, exact
messages and request template, explicit signed-int64 seed list per group, sampling
parameters, behavior policy id/revision, model id/revision, model-weight-set and
model-config SHA-256 values, export-only reward, split seed, and transport
policy. A deterministic schedule derives group ids, rollout ids, per-request
SHA-256, behavior-policy SHA-256, sampling SHA-256, and an overall schedule hash.
Request hashes cover canonical compact key-sorted UTF-8 JSON objects, not
unstable HTTP framing or header order.
Candidate `rollout_index` is schedule order; the pack's contiguous `rank` remains
the existing deterministic reward/provenance order. Both are retained, not
conflated. `model_config_sha256` hashes the exact model `config.json` bytes;
`model_sha256` uses the canonical relative-path/SHA-256/byte-length weight-file
manifest convention used by the live adapters. The orchestrator cannot derive
remote weights itself: it requires the planned values and endpoint attestation
to match, while the later live gate must independently rehash the actual loaded
model and config.

### Endpoint attestation and response rules

Every public `POST /v1/chat/completions` request is streaming and carries
`X-UmmInfer-Online-Rollout-Contract: 1` and
`X-UmmInfer-Rollout-Generation: fresh` plus batch/group/rollout/attempt,
request-hash, behavior-policy-hash/revision, model-id/revision/hash,
model-config-hash, and sampling-hash headers. A compatible endpoint must echo
all of them, with `X-UmmInfer-Behavior-Evidence: not_exposed` or `exposed`, on
**every** response, including a retryable HTTP error. Missing headers, changed
model/config/policy/sampling identities, wrong attempt or request hash, and
stale/cross-group responses fail closed. Raw chunks must also keep one unique
request id and the planned model id throughout the stream.

Only HTTP 200 `text/event-stream` responses with complete raw-SSE sidecars,
one final terminal, completed state, verified artifacts, and a finite
export-derived reward can enter an accepted bundle. Cancelled responses retain
and replay their exact raw SSE and terminal legs but are never retried or packed.
Malformed SSE/protocol responses are not retried. Timeouts, transport failures,
and explicitly listed HTTP statuses retry only when their corresponding plan
switch allows it, never beyond `max_attempts` (maximum 8), and every retry has a
new bound attempt index. Default fixture policy is no retry.

Behavior evidence is optional at the transport boundary, never synthesized. If
`not_exposed`, exported `token_ids`, `logprobs`, and `mask` are null and coverage
is `NOT_COVERED`. If `exposed`, the final SSE data frame must contain exactly one
block:

```json
"behavior": {
  "format": "umminfer-behavior-logprobs-v1",
  "tokenizer_sha256": "<64-lowercase-hex>",
  "token_ids": [11, 12, 13],
  "logprobs": [-0.125, -0.25, -0.5],
  "mask": [0, 1, 1]
}
```

The arrays must be non-empty, equal-length, and contain non-negative integer
ids, finite non-positive numeric log-probabilities, and binary integer masks with at least
one supervised token. Their canonical hash and tokenizer hash are exported.
This verifies endpoint evidence shape and raw-byte binding, not detokenization
or numerical correctness against a trainer model. A future online optimizer
must separately prove tokenizer/token/target alignment and behavior-model
identity before consuming those values.

### Accepted bundle and byte-identical replay

```text
<out>/
  plan.json
  attempts.jsonl             # diagnostic attempts; not the accepted pack input
  trajectories.jsonl         # only completed, attested, rewarded rollouts
  artifacts/                 # exact .sse sidecars and verified CAS media
  rollout_manifest.json      # online mode, hashes, counts, explicit claims
  rollout_manifest.json.sha256
  pack/                      # existing deterministic DPO/GRPO layout
    dpo_train.jsonl
    dpo_eval.jsonl
    grpo_train.jsonl         # additive online provenance on groups/members
    grpo_eval.jsonl
    manifest.json
    manifest.json.sha256
```

The pack requires exactly one row per scheduled group, exactly the planned
number of distinct rollout identities per group, no unknown/duplicate/missing
or cross-group source, exact source targets and recomputed rewards, ordered
membership-bound group ids, and recomputed baseline/std/advantages. The manifest
continues to say `REAL_TRAINING: NOT_RUN` and `optimizer: NOT_RUN`; model identity
is marked `ENDPOINT_ATTESTATION_ONLY` until the live gate independently rehashes it.

Commands are:

```text
python3 -I python/umminfer_online_rollout.py run \
  --plan configs/training/online_grpo_rollout_fixture.json \
  --endpoint 127.0.0.1:<attesting-endpoint-port> --out <new-bundle>
python3 -I python/umminfer_online_rollout.py replay \
  --input <accepted-bundle> --out <new-replay-directory>
python3 -I tests/online_rollout_test.py
```

Replay revalidates the plan, exact requests, response attestations, raw SSE/CAS,
rewards, group membership, and behavior evidence in a fresh process, then
requires byte-identical accepted JSON/JSONL, manifests, pack files, and artifact
trees. `attempts.jsonl` is diagnostic and intentionally outside that accepted
file set. The committed fixture uses only logical ticks and a deterministic fake
endpoint; it is not real generation evidence.

### Later isolated live gate

Real online generation and optimizer/quality outcomes remain **NOT_RUN**. A later
gate must use a separately approved, isolated endpoint and training process;
record actual model/policy revision, model/config/weight hashes, exact requests,
seeds and sampling; implement and verify every response attestation; capture
observed monotonic raw-SSE ticks; reject cancellation/drift; and reproduce the
accepted trajectories and groups byte-for-byte in fresh processes. It must keep
serving free of trainer/optimizer imports and the trainer free of inference
imports. If behavior log-probabilities are consumed, the gate must additionally
prove model/tokenizer/token-target/mask alignment rather than treating the
structural evidence check as sufficient. An online optimizer adapter, real
model-quality evaluation, generalized improvement, and production-model training
are separate later gates. The current public ummInfer endpoint does not emit the
required attestation or behavior-evidence headers, so it is deliberately rejected
rather than treated as compatible.

## Bounded real joint-SFT optimizer gate

`python/umminfer_joint_sft_live.py` is a narrow execution adapter around the
existing Stage C planner; it does not replace the dataset builder, span parser,
ablation resolver, or batch scheduler.  It consumes a verified SFT pack built
from schema-v2 exports and supports the deliberately limited configuration in
`configs/training/joint_sft_live_gate.json`: one local Qwen2 causal LM, full
parameters (no PEFT), BF16 eager attention, batch/accumulation/epoch counts of
one, constant LR, and zero weight decay.  It is not the production 7B training
recipe and does not execute DPO, GRPO, vision/media forward passes, or serving.

The adapter performs all model-independent validation before importing `torch`
or `transformers`. The default `--token-source legacy` retains the accepted
full-text re-encoding operation-for-operation. The opt-in
`--token-source token_export` requires every `sft_train` row to carry a
supported v1/v2 export and its complete bound source record; missing, mixed,
partial, stale, mismatched, or malformed exports fail without legacy fallback.
SFT rows without exports retain byte-identical JSONL.

A fresh tokenizer-only preflight makes Torch unavailable and blocks its import,
rehashes the complete local model/tokenizer directory, and calls the existing
strong v1/v2 validator before planner output or optimizer state is created.
Only the prompt prefix is tokenized for training. Exported target ids, raw
offsets, labels, weights, and annotations are consumed directly; a separate
legacy encoding is used only to require exact joint_full equality, never as a
fallback. Tokenizer/export identity is bound into checkpoint resume identity.
Every run records raw preflight and per-step export evidence plus canonical
gradient and FP32 parameter-update hashes. Exported post-hoc ids are never
claimed as sampled online behavior ids.

Scheduled fixtures must include both text-only and structured visual targets.
Each visual slot must bind to its reconstructed marker, verified CAS artifact,
and completed terminal leg; route headers, prompt bodies, and closing `]]`
delimiters must all receive positive target loss weight. Non-finite
loss/gradients/parameters, empty gradients, unchanged parameters, out-of-range
sequences, checkpoint identity mismatches, and hash tampering are hard failures.

Run the deterministic no-model/no-GPU preflight suite:

```text
python3 -I tests/joint_sft_live_test.py
```

Run the real gate with the caller selecting exactly one explicitly idle GPU:

```text
CUDA_VISIBLE_DEVICES=<idle-gpu> python3 -I tests/joint_sft_live_test.py --live \
  --model-path <local-qwen2.5-0.5b-instruct> \
  --evidence-dir <new-evidence-dir> \
  --source-commit <40-character-commit>
```

The live harness creates two deterministic schema-v2 fixtures through the real
recorder and dataset builder, strongly exports them with the exact local Qwen
tokenizer, then starts four independent training processes:

1. two legacy AdamW steps;
2. the same frozen pack through two export-driven AdamW steps;
3. the export run stopped after step 1 (`--stop-after 1`); and
4. a fresh process resumed from that step-1 checkpoint through step 2.

Two more processes submit re-signed target-binding and partial-export input;
both must exit 2 before Torch import or planner/preflight/checkpoint creation.

The checkpoint contains model and optimizer tensors, schedule position, and CPU
and CUDA RNG states.  Its sidecar and source/config/data/model/order identity
are rechecked before loading.  The gate compares the initial split step and the
resumed second step against the uninterrupted run, including loss, gradient
norm, record order, target/mask hashes, all-parameter hashes, and optimizer-state
hashes. Legacy and export paths must match exactly on ids, labels, masks,
losses, canonical gradients, FP32 parameter updates, and model/optimizer hashes.
Fresh-process export resume must also reproduce checkpoint bytes exactly.

Every run records the source commit and listed source-file hashes, exact and
resolved config hashes, dataset and pack hashes, all available model files and
weight hashes, seed derivation, deterministic batch/order hashes, per-step wall
time and CUDA peak allocated/reserved bytes, changed finite parameter counts,
and checkpoint hashes.  `acceptance.json` and `acceptance.json.sha256` summarize
the gate.  Evidence directories and checkpoints are runtime artifacts and are
not committed.  Model quality, generalized improvement, DPO/GRPO outcomes, and
upstream model revision verification remain `NOT_RUN` unless separately measured.

### Exported-token SM90 acceptance — 2026-11-26

Source `93214a5261ebaf2fd6108e129aa5e7993af18aa7` passed the complete gate on
explicitly idle physical GPU 7 with the local Qwen2.5-0.5B tokenizer/model,
Torch 2.6.0+cu124, Transformers 5.23.0, and tokenizers 0.23.3. Legacy and
export-driven runs both produced losses `8.003159523010254` and
`4.829397201538086`, with exact matching ids, labels, masks, gradients,
parameter updates, and model/optimizer hashes. Fresh-process export resume
reproduced both checkpoint byte hashes, and both invalid processes exited 2
without planner, preflight, checkpoint, or optimizer output. Acceptance SHA-256
is `04b38f4f535be8deb3fcb7d9589dab91c3f344fc6d91acb2fbd8cdc09642182b`.
The exact commands, all source/config/data/model/tokenizer/binary hashes,
per-step evidence, independent verification, GPU cleanup, and quality limits
are in [exported-token-sft-live](exported-token-sft-live.md).

### Historical legacy-only SM90 acceptance — 2026-11-26

The final gate ran on explicitly idle physical GPU 7 (NVIDIA H20, SM90), with
`CUDA_VISIBLE_DEVICES=7` and logical `cuda:0`.  No server, realtime stack, or
network port was started.  Python 3.12.3, torch 2.6.0+cu124, transformers
5.23.0, deterministic algorithms, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` were
recorded.  Source commit was `b2198d5593fa586c1a957b7a2a40a88b21bf27cc`.
The runtime evidence is in `/tmp/umminfer-joint-sft-live-16/final-b2198d5/`;
`acceptance.json` SHA-256 is
`384057635e2a8f308f55849066a0387c4db17891ed6eb8100946d6f567f38180`.
The evidence directory contains all per-file model hashes and runtime logs;
the following table records the headline reproduction values.

| Item | Measured value |
| --- | --- |
| Live adapter / existing trainer / trajectory source SHA-256 | `f48c217e296d0f787f01ff7dcc3564aa9b8261495abeedf16baa12b010a634e8` / `2c0405d4a3a692aae31a4cc94d50e90e169f471038bac17dae94edeacf6725fd` / `3307e3b84156b4b6effce8506c48876051f4d0f9962b7af1247a572f43b517e9` |
| Live test / dataset-builder source SHA-256 | `d7bfcaf5f869393541aca06efd93210394d34389092c2a15dbf315b1633dc40e` / `1742eb39d611f4affce14b6205b4d750fc7c242e6c4ca56a4fe1977d981bd348` |
| Live config exact / resolved SHA-256 | `b7c7e0d3503e1ed339cdc18617a7a919709c3357f4ea51992f4c8abd67479f2d` / `3c791a37d2b3cb190ecc2c86d8aeb72b4778892393817cb345347c50ee3b0c64` |
| Schema-v2 fixture JSONL / dataset manifest / SFT train pack SHA-256 | `f6e2496a2f62b5cfe13288855d10fa2c4bb19b571e921346749f3eb5ba9496fc` / `6d69f42f76f0d20d59221ff21b9c5a4c55e3dc4136400a7196e89168d2bf4d45` / `9bf15b0fb13d25c1d7a94084174d759e58351dddcf327514d969ca63d227f24e` |
| Model and available weights | Qwen2.5-0.5B-Instruct; revision `UNKNOWN`; `model.safetensors` 988,097,824 bytes, SHA-256 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`; available-weight-set SHA-256 `0d79ed576a7c2e115025787de0f0536c260affeaeb73f6c8022141290609eec7` |
| Seeds | Fixture `joint-sft-live-fixture-v1`; optimizer `umminfer-stage-c-joint-sft-live-gate-v1`; derived uint31 `1139696663` |
| Batch schedule / per-step record-order SHA-256 | `b56622bbb362b6ffd69969d3925c0161e29b59980786e25b33cd39f0957dee51` / step 1 `474aeebf7e75d747153b34ae42697e628e72e0f3f19ebdfe2e0f7490d60f8a90`, step 2 `54e19d3027ee151b167924b601d41fa551420689b0004f2bb8d682cebda06841` |

Two uninterrupted full-parameter AdamW updates used LR `1e-4`, zero weight
decay, and gradient clipping at `1.0`.  Every one of the 290 gradient tensors
was non-zero.  All post-step parameters were finite.

| Step / target | Loss | Grad norm | Changed tensors / scalar parameters | Max absolute delta | Step wall ms | CUDA peak allocated / reserved bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1: interleaved image+video, two completed terminal legs | 4.336180686950684 | 148.0 | 254 / 342,615,512 | 0.0001220703125 | 1,973.240 | 5,309,981,184 / 5,790,236,672 |
| 2: text-only | 2.019967555999756 | 200.0 | 254 / 279,815,909 | 0.0001220703125 | 1,535.902 | 5,593,522,176 / 6,880,755,712 |

The visual step masked 10 prompt tokens and supervised 34 target tokens,
including 10 route-tag tokens, 15 visual-prompt-body tokens, and two closing
terminal delimiters.  Its input/label/loss-weight SHA-256 values are
`c47b55218d0d3af48b53049388674438e9fdb5d0739f3395894134937344a3ea`,
`c917ae27541197d1fd836c83a2b5b0c7465a5396868ab51057dfa789b9a69125`,
and `12bbeb3771090f5563cb3b308668bc6f89c4e902cda6981822cccc9a4d4d3e37`.
The text step masked 9 prompt tokens and supervised 5 target tokens.  Total
uninterrupted run wall time, including hashing and checkpoint writes, was
23,104.351 ms.

Base, step-1, and final all-parameter SHA-256 values were respectively
`4b4c678c452575cbc29e0b80a86698fd07fdb83bc7c1b9710e50cfca13d06b9a`,
`7705d69170afec77691275369fc7d434b81cbcda1553232aa0d1c1b375a7beb9`, and
`0281897a6ac8bbf28c050bf23d2e1af4b6fa5efad87f26c3165335967f873b9b`.
The final optimizer-state SHA-256 was
`a09d323cdb7def4d7bc0edce142cec8bad85b692a91289706a755b4f8857ee89`.
Step-1 and step-2 checkpoint SHA-256 values were
`b15704c61e0573cc67fb7878c40096ab3ad00e234c44ed7567efb3f52db2e163` and
`9245fdd5b14aee16f59a6c0e206b49a16b473ecc7fab279f4d26f676c3cced1a`.

The independent split-first process produced the same step-1 parameter,
optimizer, and checkpoint hashes.  Its fresh-process resume produced the same
second-step loss, gradients, parameter/optimizer hashes, and step-2 checkpoint
hash as the uninterrupted run; every exact comparison field matched.  The
re-signed invalid target exited 2 before ML import or optimizer execution and
created neither a success manifest nor a checkpoint.  After all subprocesses
exited, physical GPU 7 returned to 3 MiB used and 0% utilization.  GPUs 2–6 and
the resident service were not contacted or restarted.

At that SFT gate, model quality, generalized improvement, DPO, GRPO, upstream
revision verification, and vision/media forward execution remained **`NOT_RUN`**.
The separate bounded DPO gate is documented below.

## Exported-token joint-DPO CPU data and preflight gate

`python/umminfer_joint_dpo_live.py --token-source legacy` remains the default and
preserves the accepted full-text encoding and real DPO path. Explicit
`--token-source token_export` requires a supported v1/v2 export and complete
source binding on both chosen and rejected sides of every pair. The preference
builder retains those blocks additively; unexported DPO pack bytes and all GRPO
summary shapes remain unchanged.

The DPO structural gate validates pair, prompt, provenance, source, and tokenizer
identities before planner output. A fresh Torch-isolated tokenizer process then
rehashes the exact local files/runtime and strongly validates every export. Only
the shared prompt prefix is encoded for training; exported target ids, raw
offsets, labels, weights, and annotation bindings drive both policy and reference
inputs, with legacy encoding used solely for exact equality checks. Pair/export,
packed tensor, tokenizer, and frozen-reference input identities enter checkpoint
resume identity. Invalid inputs cannot fall back on one side or cross-resume.

`--preflight-only` is the deterministic CPU-only entry point. It writes the
preflight bundle and SHA-256 with real optimizer and quality marked `NOT_RUN`.
The separate `--token-export-verify` entry point independently reconstructs and
byte-compares a complete real-local-tokenizer bundle in a fresh process without
Torch. Deterministic v1/v2 fixture subprocesses cover multilingual/NFC/emoji,
spoken and visual spans, explicit anchors, exact sigmoid/reference-mask arithmetic,
invalid-input no-output/no-ML gates, and split/resume schedule identity. Exact
commands, fixture hashes, and real-local-tokenizer evidence are in
[exported-token-dpo-preflight](exported-token-dpo-preflight.md). The real
Qwen2.5 tokenizer preflight and fresh-process verifier passed two v2 pairs and
five fail-closed negative cases with no Torch import. No GPU, model inference,
or export-driven DPO optimizer was run; model quality remains NOT_RUN.

Run the deterministic suite:

```text
python3 -I tests/joint_dpo_live_test.py
```

## Bounded real joint-DPO optimizer gate

`python/umminfer_joint_dpo_live.py` executes the existing Stage C DPO plan over
verified schema-v2-derived preference pairs. It reuses the dataset verifier,
span masker, deterministic pair scheduler, HF offset masks, tensor fingerprints,
and atomic checkpoint helpers; it does not duplicate the planner or trainer and
does not depend on a serving process. The narrow configuration in
`configs/training/joint_dpo_live_gate.json` selects one local Qwen2 causal LM,
full parameters, BF16 eager attention, one pair per step, constant LR, zero
weight decay, sigmoid loss, and a frozen independent copy of the initial base
model as reference.

The deterministic fixture builds two pairs through the real schema-v2 recorder
and preference dataset builder. Chosen and rejected responses both contain
clean prose, structured visual markers, on-disk-verified CAS artifacts, and
completed terminal-leg bindings. The gate independently audits prompt masking,
route-tag, visual-prompt-body, and closing-delimiter weights on each side. It
recomputes pair identity and margins, and rejects malformed, crossed, or
CAS-broken pairs before importing ML frameworks. Each real step records finite
loss, changed finite parameters, pre/post-update policy log-probabilities,
frozen-reference log-probabilities, weighted sums and normalizers, beta, order
and source/config/data/model/weight hashes, timing, and CUDA memory.

Default CPU preflight and opt-in live commands are:

```text
python3 -I tests/joint_dpo_live_test.py
CUDA_VISIBLE_DEVICES=<idle-gpu> python3 -I tests/joint_dpo_live_test.py --live \
  --model-path <local-qwen2.5-0.5b-instruct> \
  --evidence-dir <new-evidence-dir> \
  --source-commit <40-character-commit>
```

The live harness compares two uninterrupted steps with a split-first process and
a fresh-process resume, including exact model/optimizer/checkpoint hashes and
all DPO log-probability evidence. At that DPO gate, model quality, generalized
improvement, production-model training, GRPO, upstream revision verification,
and vision or media forward execution remained **`NOT_RUN`**. The separate
bounded GRPO gate is documented below.

### Measured SM90 DPO acceptance — 2026-11-26

The gate ran on explicitly idle physical GPU 7 (NVIDIA H20, SM90), with
`CUDA_VISIBLE_DEVICES=7`, exactly one visible logical `cuda:0`, deterministic
algorithms, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Python 3.12.3, torch
2.6.0+cu124, and Transformers 5.23.0 were recorded. No server or network port
was started or contacted. Source commit was
`8978945ba2e98feaa309b28ae02c841cfd0ac051`. Runtime evidence is under
`/tmp/umminfer-joint-dpo-live-19/evidence/`; `acceptance.json` SHA-256 is
`84e61247e182a5c08ebd7c23300182f686ef792eb20efa3f3dd80d357e1f9bc0`.

| Item | Measured value |
| --- | --- |
| DPO adapter / SFT helper / existing trainer source SHA-256 | `6b72be4f7417562d5769ebc46943287ad28127a02c3fd27b12ac2cc88022fa77` / `f48c217e296d0f787f01ff7dcc3564aa9b8261495abeedf16baa12b010a634e8` / `2c0405d4a3a692aae31a4cc94d50e90e169f471038bac17dae94edeacf6725fd` |
| Live test / trajectory / dataset-builder source SHA-256 | `194dcab6e9d6b3e97cdd446baf632c25cd2028de3173b22820eb1451366a415d` / `3307e3b84156b4b6effce8506c48876051f4d0f9962b7af1247a572f43b517e9` / `1742eb39d611f4affce14b6205b4d750fc7c242e6c4ca56a4fe1977d981bd348` |
| Exact config / resolved config SHA-256 | `ca1cfe51e2d25bd5a59546362fc719bf2ff19ed54c79d0d81880eea1c64c3b04` / `497a31c6e056ef5ff45f8573a03c671cec5d509be0ea6f39929314d74a2dd72d` |
| Schema-v2 fixture / dataset manifest / two-pair DPO train SHA-256 | `c885de192982adc68d896eab4ded3c8325b1cf2cc2f004ebf5d2a71f6b73af9d` / `26945bac05ce1edf9a64ec2263f04030b3263696717b6053488b2c28cbb0ed6f` / `2e7e15b5143bdba21cb597371849c35b911c07ba03b647ad8b184491617e4ead` |
| Model and available weights | Qwen2.5-0.5B-Instruct; revision `UNKNOWN`; `model.safetensors` 988,097,824 bytes, SHA-256 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`; available-weight-set SHA-256 `0d79ed576a7c2e115025787de0f0536c260affeaeb73f6c8022141290609eec7` |
| Seeds | Fixture `joint-dpo-live-fixture-v1`; optimizer `umminfer-stage-c-joint-dpo-live-gate-v1`; derived uint31 `999449239` |
| Schedule / pair order / per-step order SHA-256 | `ad6b4a4f599f79cfd207936881456f20f9d307018ebf730a757a15f41dbe173b` / `1d3d2f83d826df9eb0898be1636e5489b184f34fc28aaef1a7d0948a103611d4` / step 1 `7f169f3cabff625e6778d73070c233455574f47ffa9f1d0033f9cf727e51a7b4`, step 2 `e610627585394a1e2973509f4954c97e95eea9c0465e15d9bcd1d6c4e02a7109` |
| DPO reference | Frozen independent copy of the initial base model, beta `0.1`, sigmoid loss, zero label smoothing, span-weighted target-mean log-probabilities; base/reference all-parameter SHA-256 `4b4c678c452575cbc29e0b80a86698fd07fdb83bc7c1b9710e50cfca13d06b9a` |

Two uninterrupted full-parameter AdamW updates used LR `1e-4`, zero weight
decay, and gradient clipping at `1.0`. Every one of the 290 gradient tensors
was non-zero; all post-step parameters and metrics were finite.

| Step / pair | Pre-update loss | Grad norm | Changed tensors / scalar parameters | Max absolute delta | Step wall ms | CUDA peak allocated / reserved bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1: lantern chosen image+video / rejected image | 0.6931471824645996 | 6.15625 | 254 / 338,555,481 | 0.0001220703125 | 1,939.826 | 6,270,667,776 / 6,803,161,088 |
| 2: accretion-disk chosen image+video / rejected image | 0.613104522228241 | 12.4375 | 253 / 280,286,965 | 0.0001220703125 | 1,734.196 | 6,691,628,032 / 7,948,206,080 |

At the initial step, policy and reference log-probabilities matched exactly on
both sides: chosen `-4.336180686950684`, rejected `-5.1937127113342285`, implicit
margin `0`, and loss `log(2)` within FP32 evaluation. After step 1, chosen and
rejected policy log-probabilities were `-8.367234230041504` and
`-17.169437408447266`; the scaled margin increased by `0.7944671154022217` and
post-update loss was `0.3728192778466993`. Step 2 started with policy/reference
chosen log-probabilities `-7.120877265930176 / -4.153624057769775` and rejected
`-9.917153358459473 / -5.279360771179199`; its scaled margin increased by
`0.7519716262817383`, with post-update loss `0.3356916188548742`. These are
optimizer-direction measurements on two fixture pairs, not quality improvement.

Step-1 prompt-masked/supervised tokens were `10/34` chosen and `10/24` rejected;
route-tag/prompt-body/closing-delimiter tokens were `10/15/2` and `5/7/1`.
Step-2 prompt-masked/supervised tokens were `12/39` chosen and `12/26` rejected;
route-tag/prompt-body/closing-delimiter tokens were `10/16/2` and `5/7/1`.
Every side had verified CAS-backed completed terminal legs. Total uninterrupted
run wall time, including hashing, post-update evaluation, and checkpoint writes,
was `26,523.731 ms`.

The final all-parameter SHA-256 was
`9f76b885373f6ea07103b21d85af0b1d9f727b7463e02dcc99053d2d095e813a`; final
optimizer-state SHA-256 was
`bf487e7bc4dd1739fc0e8f726b5cb6cf65d1a8f074c266b0218a81694e564dda`.
Step-1 and step-2 checkpoint SHA-256 values were
`148fb996938fd34ace37e55d5c6db055d057d5a543bc518c95436e09c0d64368` and
`d68bf3cf5609e24e97c9b630c7f1dc22a79fcfe50ffaf2f19ed78a9e6622b5c0`.
The split-first process reproduced step 1 exactly. A fresh process resumed its
checkpoint and reproduced step 2 exactly: all compared losses, gradients,
policy/reference log-probability blocks, masks, parameters, optimizer state, and
both checkpoint hashes matched with zero mismatch fields. Re-signed malformed
and crossed-pair processes each exited 2, emitted no ML-import marker, and
created neither a success manifest nor a checkpoint.

After all subprocesses exited, physical GPU 7 returned to 3 MiB used and 0%
utilization, with no compute processes reported. GPUs 2–6 and the resident
service were not contacted or restarted. The prior real SFT evidence is
preserved unchanged. At that DPO run, GRPO, model quality, generalized
improvement, production-model training, upstream revision verification, and
vision/media forward execution remained **`NOT_RUN`**.

## Bounded real joint-GRPO optimizer gate

`python/umminfer_joint_grpo_live.py` executes the existing Stage C GRPO plan over
complete deterministic groups derived from schema-v2 exports. It reuses the
dataset verifier and reward interface, whole-group scheduler, DPO/SFT response
and HF-mask helpers, tensor fingerprints, and atomic checkpoint helpers. It does
not replace the planner or trainer and never starts or contacts a serving
process. The narrow configuration in
`configs/training/joint_grpo_live_gate.json` selects one local Qwen2 causal LM,
full parameters, BF16 eager attention, one whole group per step, constant LR,
zero weight decay, `clip_eps=0.2`, `kl_coef=0.04`, and a frozen independent copy
of the initial base as both old/behavior policy and KL reference.

The objective matches the existing Stage C sequence-level GRPO numerical
reference. For each rollout, a span-weighted target-mean log-probability gives
`ratio = exp(clamp(policy_logp - old_logp, -20, 20))`; the clipped surrogate is
averaged over rollouts and combined with the non-negative Schulman KL
approximation. The old policy remains the frozen initial base across this
bounded offline gate; it is not silently refreshed after step 1. Cached
old/reference log-probabilities and the behavior-model fingerprint are saved in
each checkpoint and recomputed for comparison before resume.

The fixture contains two prompts and exactly three distinct rollouts per prompt:
image+video, image-only, and two redundant image slots. Every rollout contains
clean prose and structured visual markers with completed terminal legs and real
on-disk CAS artifacts. Before any ML import, the adapter:

- rehashes the exact source JSONL inputs against the dataset manifest and
  independently revalidates every schema-v2 event, terminal, hash, and CAS byte;
- recomputes `NarrativeRoutingReward` from each export and records its set-F1,
  redundancy, expected/predicted-modality, and source-terminal components;
- requires one-to-one source/pack membership, distinct targets and provenance,
  exact configured group size, contiguous ranks, deterministic score/provenance
  order, and a group ID bound to ordered target hashes;
- recomputes the population baseline/std and normalized advantages, rejects a
  zero-variance scheduled group, and binds the trained prompt text to the source;
- validates every source record is consumed exactly once across the GRPO packs,
  then applies the existing prompt/route/prompt-body/terminal HF masks.

Default CPU preflight and opt-in live commands are:

```text
python3 -I tests/joint_grpo_live_test.py
CUDA_VISIBLE_DEVICES=<idle-sm90-gpu> python3 -I tests/joint_grpo_live_test.py --live \
  --model-path <local-qwen2.5-0.5b-instruct> \
  --evidence-dir <new-evidence-dir> \
  --source-commit <40-character-commit>
```

The live harness runs two uninterrupted steps, an independent split-first
process stopped after step 1, and a fresh process resumed into a separate output
directory. It compares losses, gradients, ordered groups/rollouts, all reward and
mask audits, policy ratios/KL, parameter/optimizer/behavior hashes, and both
checkpoint byte hashes. Six additional re-signed-pack processes cover invalid
groups, missing and duplicate rollouts, numeric NaN rewards, tampered finite
rewards, and terminal/hash inconsistencies; each must fail before the ML-import
marker and create neither a success manifest nor a checkpoint.

### Measured SM90 GRPO acceptance — 2026-11-26

The gate ran on explicitly idle physical GPU 7 (NVIDIA H20, SM90), with
`CUDA_VISIBLE_DEVICES=7`, exactly one visible logical `cuda:0`, device capability
`(9, 0)`, deterministic algorithms, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
Python 3.12.3, torch 2.6.0+cu124, and Transformers 5.23.0 were recorded. No
server or network port was started or contacted. Source commit was
`19e1fba72d3ebadd7df196910175438f872e389b`. Runtime evidence is under
`/tmp/umminfer-joint-grpo-live-21/evidence/`; `acceptance.json` SHA-256 is
`480a7b4708fb023c10a12cac987daabcc456a8c2c498eed4229aa5597c2dc158`.

| Item | Measured value |
| --- | --- |
| GRPO adapter / DPO helper / SFT helper / existing trainer SHA-256 | `a34e775f1bd5d104b662ecdf70ac062a4bdc0fb1365b6a94c05c4173ab06a691` / `6b72be4f7417562d5769ebc46943287ad28127a02c3fd27b12ac2cc88022fa77` / `f48c217e296d0f787f01ff7dcc3564aa9b8261495abeedf16baa12b010a634e8` / `2c0405d4a3a692aae31a4cc94d50e90e169f471038bac17dae94edeacf6725fd` |
| Live test / dataset builder / trajectory / reward-metric SHA-256 | `1bf1a7559a1aea88c2880007326c3c396bbc0e534ebab2f60b44c99b5ef38163` / `1742eb39d611f4affce14b6205b4d750fc7c242e6c4ca56a4fe1977d981bd348` / `3307e3b84156b4b6effce8506c48876051f4d0f9962b7af1247a572f43b517e9` / `9b5e387933aad5c3a6252178562185bd3d9d0e07bf15e62b494432e7ee9eb5a4` |
| Exact config / resolved config SHA-256 | `5fe53f6b7aa29ca9c7389ee00f8819978aa50b977f1c5d9c4c2f46828f1a3ce4` / `63938049b83918b1c3ff977a65bab42fad9558700596cba28a8878151c8e2a32` |
| Schema-v2 fixture / dataset manifest / two-group GRPO train SHA-256 | `0458c07e095c6210bc4c33cd297fc94cecabf921819de702227e17e1ac475a65` / `66f8972c7ed429f67a020b4982f6bbfb4b811ab82453444a51dff419cd3da067` / `e24f024493c78f8bc8628a971a3aaae7d34a1942715b99882ef85d95b2d010aa` |
| Model and available weights | Qwen2.5-0.5B-Instruct; revision `UNKNOWN`; `model.safetensors` 988,097,824 bytes, SHA-256 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`; available-weight-set SHA-256 `0d79ed576a7c2e115025787de0f0536c260affeaeb73f6c8022141290609eec7` |
| Seeds | Fixture `joint-grpo-live-fixture-v1`; optimizer `umminfer-stage-c-joint-grpo-live-gate-v1`; derived uint31 `960949850` |
| Schedule / group-order SHA-256 | `4e728b95f0073346802f3dba8a12ae611a76d4d4494ab8cf284dfe3b2229ecec` / `c492fd2a2b5691477b5b5510c7fb4a02c7f4d7f56feaf2beb3edc0d822d77371` |
| Objective | Frozen initial base as old+reference; weighted-target-mean sequence ratio; rollout-mean clipped surrogate; clip `0.2`; KL coefficient `0.04`; log-ratio/KL clamp `20.0` |

Each group's export-derived rank-0/1/2 rewards were respectively `1.0`,
`0.6666666666666666`, and `0.16666666666666663`. Their F1 components were
`1.0`, `2/3`, and `2/3`; redundant-slot rates were `0`, `0`, and `0.5`.
Population baseline/std were `0.611111111111111 / 0.3424674446093876`, and
ordered normalized advantages were
`[1.135549947915338, 0.1622214211307626, -1.2977713690461001]`.

| Step | Group ID | Ordered rollout IDs (best / partial / redundant) | Group / membership order SHA-256 |
| --- | --- | --- | --- |
| 1 | `ef44963f41c433592613107f8354dd74` | `66568b201c2f9261659e6594fffcb21b` / `17f4942856a627b83b9579d0f9f91906` / `960df4e260034a056f833db5a2aa85c7` | `b8c4df6c109a237d5dfb759f7f3bc7632d0cc158be71599993099b2a2c135bf8` / `851d60020d99d1a06aaac41da6437d5543e8f432dd9cd6d75cf188aa0af4c7d2` |
| 2 | `039e01981969c6e2164f50b79bb33ee6` | `3b2137f2f23de9fe7bfd0f686856c226` / `481f6ae6a0cd642163f26c106f315e07` / `7f33acb951aaa282795ef3f0b198e572` | `3cfba510424c80841271e39da17fb2f71cd4a04d7060516138aad599a287b3f5` / `6af27d6779d52106eba78e2b23beeb2c67b95082090290c2b84e65101e3a1931` |

Two uninterrupted full-parameter AdamW updates used LR `1e-4`, zero weight
decay, and gradient clipping at `1.0`. Every one of the 290 gradient tensors was
non-zero on both steps; all metrics and post-step parameters were finite.

| Step | Pre-update loss | Post-update loss | Grad norm | Changed tensors / scalar parameters | Max absolute delta | Step wall ms | CUDA peak allocated / reserved bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.967053879312289e-09 | 1.2877223492189325 | 32.75 | 254 / 341,486,883 | 0.0001220703125 | 2,080.076 | 6,269,683,200 / 6,889,144,320 |
| 2 | 0.44793424010276794 | 0.12280015869104227 | 88.5 | 252 / 281,449,704 | 0.0001220703125 | 1,893.023 | 6,781,109,760 / 8,057,257,984 |

| Step | Mean policy ratio pre → post | Mean KL pre → post | Clip fraction pre → post |
| --- | ---: | ---: | ---: |
| 1 | `1.0 → 0.41552761728517856` | `0.0 → 31.783343393459358` | `0.0 → 0.3333333333333333` |
| 2 | `0.6836801035517396 → 3.9505230142849093` | `11.938733363059441 → 0.580464061061047` | `0.6666666666666666 → 0.6666666666666666` |

The first pre-update loss is approximately zero because normalized advantages
sum to zero at unit ratios and zero KL; its non-zero real gradients still produce
a measured parameter update. The first post-update objective increases, while
the second decreases. These are recorded optimizer mechanics on two static
fixture groups, not a monotonic-training or model-quality claim. Total
uninterrupted wall time, including behavior-logp caching, post-update evaluation,
hashing, and checkpoint writes, was `27,146.972 ms`.

Step-1 best/partial/redundant prompt-masked/supervised tokens were `10/34`,
`10/25`, and `10/39`; route-tag/prompt-body/closing-delimiter tokens were
`10/15/2`, `5/10/1`, and `10/20/2`. Step-2 corresponding supervised counts were
`12/39`, `12/30`, and `12/47`; structured counts were `10/16/2`, `5/12/1`, and
`10/24/2`. The full evidence includes input-ID, label, loss-weight, and
membership hashes for every rollout.

Base, step-1, and final all-parameter SHA-256 values were respectively
`4b4c678c452575cbc29e0b80a86698fd07fdb83bc7c1b9710e50cfca13d06b9a`,
`e969633afce2ffcf2d3cd0bccdc98be1172815457a93c7c6773f8c0234ee849f`, and
`b1ce0dd8e96b780142de33b5016049f95ac1a8108891ad55e361f16dff002d18`.
The behavior/reference fingerprint remained the base hash throughout. Final
optimizer-state SHA-256 was
`fb1924934bf43ca1d95a8af943a9cbad381b033cbc9fdb3832256bbcc5a421a5`.
Step-1 and step-2 checkpoint SHA-256 values were
`a519e5364f1dc27fa88849903da1fa72854d3588ba4ced11ca6323934191858a` and
`e0bffea20e4fb375322584aff026e9ac714f156295ce8be041c9ec1e343d7806`.

The split-first process reproduced step 1 and its checkpoint exactly. Its fresh
process revalidated the cached old/reference log-probabilities, resumed step 2,
and reproduced every compared field and the second checkpoint with zero mismatch
fields. Final parameter and optimizer hashes matched the uninterrupted run. All
six rejection processes exited 2 before ML import or optimizer execution and
created neither a success manifest nor a checkpoint. CPU preflights additionally
cover reordered rollouts, infinite advantages, advantage/prompt/target tampering,
and actual source-export/CAS byte changes.

After all subprocesses exited, physical GPU 7 returned to 3 MiB used and 0%
utilization with no compute processes. No training work was submitted to GPUs
0/1 or 2–6, and the resident service was not contacted or restarted. Prior real
SFT and DPO evidence is preserved unchanged. Online rollout generation, model
quality, generalized improvement, production-model training, upstream revision
verification, and vision/media forward execution remain **`NOT_RUN`**.
