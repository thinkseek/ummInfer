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

## Verification & Claim Boundary

Run the offline CPU-only contract tests:

```text
python3 -I tests/sft_dataset_test.py
```

| Capability | Status | Notes |
| --- | --- | --- |
| Offline v1/v2 trajectory integrity & CAS audit | **COVERED** | Deterministic CPU-only unit tests |
| SFT interleaved beat-marker reconstruction & dedup | **COVERED** | Byte-identical across repeated runs |
| Export-only DPO pair & GRPO group-advantage packing | **COVERED** | Pluggable `load_reward("module:attr")` |
| Zero train/eval prompt leakage & manifest sidecar | **COVERED** | Verified at build time and `--audit-only` |
| Real SFT / DPO / GRPO GPU training runs | **`NOT_RUN`** | No optimizer step or model weight update claimed |
