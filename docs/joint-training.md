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

## Stage C Live-to-Training E2E Pipeline Driver (`tools/umminfer_stage_c_pipeline.py`)

`tools/umminfer_stage_c_pipeline.py` bridges exported `umminfer_rl_trajectory`
schema-v1/v2 JSONL trajectories directly through `python/umminfer_sft_dataset.py`
and `python/umminfer_joint_trainer.py` in a single content-addressed, stdlib-only
pass (100% decoupled from C++ serving code under `src/`):

1. **Strict trajectory & CAS validation**: fails closed on malformed JSON,
   unsupported `schema_version`, tampered text/event digests, or missing/tampered
   content-addressed media artifacts.
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
per prompt) to `POST /v1/chat/completions` (`stream: true`), parses the live SSE
stream into verified schema-v2 JSONL records with local CAS media artifacts, and
executes the identical SFT -> Preference -> Leakage Audit -> Joint Trainer
pipeline end-to-end (`source_mode: "live_capture"`, `REAL_TRAINING: "NOT_RUN"`).
The bounded real SFT and DPO optimizer adapters below are separate opt-in
commands and do not contact this capture endpoint:

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
python3 -I tests/joint_sft_live_test.py
python3 -I tests/joint_dpo_live_test.py
```

| Capability | Status | Notes |
| --- | --- | --- |
| Offline v1/v2 trajectory integrity & CAS audit | **COVERED** | Deterministic CPU-only unit tests |
| SFT interleaved beat-marker reconstruction & dedup | **COVERED** | Byte-identical across repeated runs |
| Export-only DPO pair & GRPO group-advantage packing | **COVERED** | Pluggable `load_reward("module:attr")` |
| Zero train/eval prompt leakage & manifest sidecar | **COVERED** | Verified at build time and `--audit-only` |
| End-to-end trajectory -> pack -> ablation-matrix dry-run driver | **COVERED** | `tools/umminfer_stage_c_pipeline.py` + `tests/stage_c_pipeline_test.py` |
| Real joint-SFT optimizer gate | **PASS (SM90)** | Two real AdamW steps, changed finite parameters, exact resume equivalence, and invalid-input rejection; see measured record below |
| Real joint-DPO optimizer gate | **PASS (SM90)** | Two real sigmoid-DPO AdamW steps, policy/reference log-probabilities, independent chosen/rejected masks, exact resume equivalence, and pre-import malformed/crossed-pair rejection |
| Real GRPO GPU training run | **`NOT_RUN`** | No group-relative preference optimizer step or model weight update claimed |

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
or `transformers`.  It then uses an offline fast tokenizer's character offsets
to mask every prompt token and assign the existing Stage C span weights to each
assistant token.  Scheduled fixtures must include both text-only and structured
visual targets.  Each visual slot must bind to its reconstructed marker,
verified CAS artifact, and completed terminal leg; route headers, prompt bodies,
and closing `]]` delimiters must all receive positive target loss weight.
Non-finite loss/gradients/parameters, empty gradients, unchanged parameters,
out-of-range sequences, checkpoint identity mismatches, and hash tampering are
hard failures.

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
recorder and dataset builder, then starts three independent training processes:

1. two uninterrupted AdamW steps;
2. the same two-step run identity stopped after step 1 (`--stop-after 1`); and
3. a fresh process resumed from that step-1 checkpoint through step 2.

The checkpoint contains model and optimizer tensors, schedule position, and CPU
and CUDA RNG states.  Its sidecar and source/config/data/model/order identity
are rechecked before loading.  The gate compares the initial split step and the
resumed second step against the uninterrupted run, including loss, gradient
norm, record order, target/mask hashes, all-parameter hashes, and optimizer-state
hashes.  A fourth process submits a re-signed but semantically invalid target;
it must exit nonzero without a checkpoint or success manifest.

Every run records the source commit and listed source-file hashes, exact and
resolved config hashes, dataset and pack hashes, all available model files and
weight hashes, seed derivation, deterministic batch/order hashes, per-step wall
time and CUDA peak allocated/reserved bytes, changed finite parameter counts,
and checkpoint hashes.  `acceptance.json` and `acceptance.json.sha256` summarize
the gate.  Evidence directories and checkpoints are runtime artifacts and are
not committed.  Model quality, generalized improvement, DPO/GRPO outcomes, and
upstream model revision verification remain `NOT_RUN` unless separately measured.

### Measured SM90 acceptance — 2026-11-26

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

## Bounded real joint-DPO optimizer gate

`python/umminfer_joint_dpo_live.py` executes the existing Stage C DPO plan over
verified schema-v2-derived preference pairs.  It reuses the dataset verifier,
span masker, deterministic pair scheduler, HF offset masks, tensor fingerprints,
and atomic checkpoint helpers; it does not duplicate the planner or trainer and
does not depend on a serving process.  The narrow configuration in
`configs/training/joint_dpo_live_gate.json` selects one local Qwen2 causal LM,
full parameters, BF16 eager attention, one pair per step, constant LR, zero
weight decay, sigmoid loss, and a frozen independent copy of the initial base
model as reference.

The deterministic fixture builds two pairs through the real schema-v2 recorder
and preference dataset builder.  Chosen and rejected responses both contain
clean prose, structured visual markers, on-disk-verified CAS artifacts, and
completed terminal-leg bindings.  The gate independently audits prompt masking,
route-tag, visual-prompt-body, and closing-delimiter weights on each side.  It
recomputes pair identity and margins, and rejects malformed, crossed, or
CAS-broken pairs before importing ML frameworks.  Each real step records finite
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
all DPO log-probability evidence.  Model quality, generalized improvement,
production-model training, GRPO, upstream revision verification, and vision or
media forward execution remain **`NOT_RUN`**.

### Measured SM90 DPO acceptance — 2026-11-26

The gate ran on explicitly idle physical GPU 7 (NVIDIA H20, SM90), with
`CUDA_VISIBLE_DEVICES=7`, exactly one visible logical `cuda:0`, deterministic
algorithms, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.  Python 3.12.3, torch
2.6.0+cu124, and Transformers 5.23.0 were recorded.  No server or network port
was started or contacted.  Source commit was
`8978945ba2e98feaa309b28ae02c841cfd0ac051`.  Runtime evidence is under
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
decay, and gradient clipping at `1.0`.  Every one of the 290 gradient tensors
was non-zero; all post-step parameters and metrics were finite.

| Step / pair | Pre-update loss | Grad norm | Changed tensors / scalar parameters | Max absolute delta | Step wall ms | CUDA peak allocated / reserved bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1: lantern chosen image+video / rejected image | 0.6931471824645996 | 6.15625 | 254 / 338,555,481 | 0.0001220703125 | 1,939.826 | 6,270,667,776 / 6,803,161,088 |
| 2: accretion-disk chosen image+video / rejected image | 0.613104522228241 | 12.4375 | 253 / 280,286,965 | 0.0001220703125 | 1,734.196 | 6,691,628,032 / 7,948,206,080 |

At the initial step, policy and reference log-probabilities matched exactly on
both sides: chosen `-4.336180686950684`, rejected `-5.1937127113342285`, implicit
margin `0`, and loss `log(2)` within FP32 evaluation.  After step 1, chosen and
rejected policy log-probabilities were `-8.367234230041504` and
`-17.169437408447266`; the scaled margin increased by `0.7944671154022217` and
post-update loss was `0.3728192778466993`.  Step 2 started with policy/reference
chosen log-probabilities `-7.120877265930176 / -4.153624057769775` and rejected
`-9.917153358459473 / -5.279360771179199`; its scaled margin increased by
`0.7519716262817383`, with post-update loss `0.3356916188548742`.  These are
optimizer-direction measurements on two fixture pairs, not quality improvement.

Step-1 prompt-masked/supervised tokens were `10/34` chosen and `10/24` rejected;
route-tag/prompt-body/closing-delimiter tokens were `10/15/2` and `5/7/1`.
Step-2 prompt-masked/supervised tokens were `12/39` chosen and `12/26` rejected;
route-tag/prompt-body/closing-delimiter tokens were `10/16/2` and `5/7/1`.
Every side had verified CAS-backed completed terminal legs.  Total uninterrupted
run wall time, including hashing, post-update evaluation, and checkpoint writes,
was `26,523.731 ms`.

The final all-parameter SHA-256 was
`9f76b885373f6ea07103b21d85af0b1d9f727b7463e02dcc99053d2d095e813a`; final
optimizer-state SHA-256 was
`bf487e7bc4dd1739fc0e8f726b5cb6cf65d1a8f074c266b0218a81694e564dda`.
Step-1 and step-2 checkpoint SHA-256 values were
`148fb996938fd34ace37e55d5c6db055d057d5a543bc518c95436e09c0d64368` and
`d68bf3cf5609e24e97c9b630c7f1dc22a79fcfe50ffaf2f19ed78a9e6622b5c0`.
The split-first process reproduced step 1 exactly.  A fresh process resumed its
checkpoint and reproduced step 2 exactly: all compared losses, gradients,
policy/reference log-probability blocks, masks, parameters, optimizer state, and
both checkpoint hashes matched with zero mismatch fields.  Re-signed malformed
and crossed-pair processes each exited 2, emitted no ML-import marker, and
created neither a success manifest nor a checkpoint.

After all subprocesses exited, physical GPU 7 returned to 3 MiB used and 0%
utilization, with no compute processes reported.  GPUs 2–6 and the resident
service were not contacted or restarted.  The prior real SFT evidence is
preserved unchanged.  GRPO, model quality, generalized improvement,
production-model training, upstream revision verification, and vision/media
forward execution remain **`NOT_RUN`**.
