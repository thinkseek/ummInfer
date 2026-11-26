# Real Qwen3.8 online-GRPO optimizer and exact-resume gate

Date: 2026-11-27. Final integration base: official Wave 89
`0ab22bc56e6c51d447a13ad59aef775c364f5126`, which contains accepted Task #60
through Wave 87 and Task #62/#65. The original execution chain was
`d12a6d729b8ef774308784193614e7bf430a32c6`,
`0b5b4baeedbb628e6dfdf395ef58321fbedfd893`,
`0e8291863a594e19f527c51bf5de0b3214a37301`,
`23ed886f4429139bde83c51e1b201da94f9f3041`,
`6b1bf9651b2546ce6a61deeb4163ba8af95c3eda`,
`81d94af85cdb3e176ed50bf4194d2774c6b5073c`,
`d71072dd0f2195ca5c91fbb970ec8f717f07db98`, and
`1603fa91525a90b07d0f1f16541151f55a960436`. The corresponding Wave 89 rebased
chain is `41e9f0c861175084c7fa674619f767d9f9e9aa68`,
`16fa34a17418ce5a437e73db23eaae0517ce943c`,
`2d30faf452d4e3d8907ea43cb44486ad75c90f91`,
`92856b6ce876d8932cbc0ee788749df13a9d661c`,
`6a0f5bb09d3c391a90385320f58bb50f6a49f319`,
`fceb68a4cfa02845920e6da3f95022e0bd8ca00c`,
`9762862802c5faa103043b5ba43348add0b45e17`, and
`ee9f04be27288c43811a3022e317535c3f41dba4`.

This document covers optimizer mechanics and the blocked real gate only. It does
not claim a model-quality gain, generalized training improvement, production
deployment, or a new native serving capability.

## Measured outcome — real optimizer `NOT_RUN`

The frozen final public plan at execution commit `1603fa9` produced six fresh,
attested, `COVERED` text-only rollouts with 43 sampled tokens and 43 supervised
masks. It did **not** satisfy the required two nonzero-variance groups:

| Dataset `group_id` | Prompt | Ordered rewards | Population std | Optimizer eligibility |
| --- | --- | --- | --- | --- |
| `f22f11813e1e5b0546f3ab9ab2b3a2b8` | `8642×7351=63527342` | `[1,1,0]` | `0.4714045207910317` | Nonzero, but cannot alone satisfy the two-group gate |
| `c75b9d21120dce7d7755e7e92905c54e` | `6287×5399=33943513` | `[0,0,0]` | `0.0` | Excluded; zero variance |

The complete batch was therefore rejected before native replay, token export,
preflight, model training, or optimizer execution. No offline continuation,
subgroup repetition, reward relabeling, or additional prompt/sampling mutation is
permitted. Actual gradients, parameter deltas, real checkpoints, and
uninterrupted/resumed second updates are all **`NOT_RUN`**. The configured LoRA
surface remains 17,408 parameters, but that is not a measured real-model update.

The independent stdlib auditor rebuilt the raw R6 plan, requests, attestation,
SSE/CAS, sampled evidence, rewards, groups, and pack. R6 plan SHA-256 is
`a05c3004f4732fc45ba7dae7907faa6de6b14ab89648717da0e4ea23c14c1c51`, schedule
SHA-256 is `e1fd317832a1f6fe79c398f81955471c4ba9ffb3271b3e2db1a355650ffe408d`,
sampling SHA-256 is
`ed8f05ad52e633737d5954f5446892383beab52d64db4ae26a8e72196be6d64a`, and the
raw source file-set SHA-256 is
`01bd8c7261f422c2800a0eb4bf792f58b41829eba4bfc944e9c9cd2997822047`.
The full attempt checksums and all earlier failures are sealed in
[online-grpo-optimizer-sm90.json](evidence/online-grpo-optimizer-sm90.json).

## Boundary and frozen contracts

The accepted frozen-offline GRPO adapter remains unchanged and continues to
reject online groups. `python/umminfer_online_grpo_optimizer.py` is a separate
adapter over the accepted online rollout, attestation, token-export, preflight,
and independent-audit contracts. It never starts or contacts a server and has no
endpoint argument.

The real gate uses only `Qwen/Qwen3.8-27B` (`qwen3_5` /
`Qwen3_5ForConditionalGeneration`), with the same exact 18-shard weight manifest,
configuration, and loaded tokenizer identity as the accepted native behavior
rollouts. Upstream revision remains `UNKNOWN`; file hashes, rather than a revision
label, define the compared model. No tiny or substitute model is optimizer
evidence.

Two fresh public groups each contain three uniquely identified, seeded rollouts.
The plan uses nonzero-temperature sampling and the export-only
`VerifiableArithmeticReward`: strict integer-only answers to
`6287×5399=33943513` and `8642×7351=63527342` in the final plan (earlier retained
products remain supported for replay). Both producer and independent auditor
implement the same frozen mapping. Every scheduled response must be
fresh, attested, completed with
`finish_reason=stop`, raw-SSE/CAS backed, and `COVERED` with actual sampled ids,
masks, and log-probabilities. Both groups must have nonzero population reward
variance; the accepted zero-variance Task #60 groups are explicitly rejected and
cannot become optimizer evidence.

Every accepted record receives a fresh format-v2 export with the exact local fast
tokenizer. Strong source-span/mask validation, sampled/raw-target id equality,
EOS/unknown-token exclusion, byte-identical replay, and the independent online
evidence audit run before optimization. Native-rendered prompt ids come from the
independent replay binary, not post-hoc clean-context retokenization.

## Bounded trainable surface and objective

Only one rank-1 LoRA is installed at
`model.language_model.layers.3.self_attn.q_proj`:

| Item | Value |
| --- | --- |
| Base / vision / MTP weights | Frozen BF16, no gradient, no optimizer membership |
| LoRA A / B | FP32, shapes `[1,5120]` and `[12288,1]` |
| Trainable parameters | **17,408** in two tensors |
| Adapter initialization | Seeded A, exactly zero B; initial policy must equal the adapter-disabled reference |
| Attention / cache | Eager, `use_cache=False`, non-reentrant gradient checkpointing |
| Optimizer | AdamW, LR `1e-3`, betas `(0.9,0.999)`, epsilon `1e-8`, zero weight decay, `foreach=False` |
| Gradient bound | Norm clipping at `1.0`, with preclip and postclip raw tensors retained |
| Updates | Exactly two, one complete three-rollout train group per update |

For each member, causal packing is `prompt + sampled_response`. Prompt and
unsupervised response labels are excluded. Only the necessary response-plus-one
LM-head rows are computed; dropping the final row implements the causal shift.
Current log-probabilities are FP32 log-softmax selections reduced in FP64. The
sampled native behavior remains fixed. The reference is the same frozen base with
all adapters disabled under `no_grad`; its raw per-token cache is recomputed and
checked again on resume.

The objective is the accepted online clipped sequence surrogate plus a separate
reference Schulman KL:

```text
ratio = exp(clamp(mean(current) - mean(sampled_behavior), -C, C))
surrogate = min(ratio * advantage,
                clamp(ratio, 1-epsilon, 1+epsilon) * advantage)
d_kl = clamp(mean(reference) - mean(current), -C, C)
KL = exp(d_kl) - d_kl - 1
loss = -surrogate + kl_coef * KL
```

`clip_eps=0.2`, `kl_coef=0.04`, and `C=10`. Losses use masked response means
followed by a rollout mean. Backward runs separately per rollout at `1/group_size`
scale, which is the same bounded whole-group objective without retaining all
rollout graphs simultaneously. Native-versus-HF initial equality is not assumed:
only repeated/resumed HF execution is required to be byte-identical.

The batch is generated once from the frozen behavior policy. The second update
reuses that online-origin batch and is therefore off-policy relative to the
updated adapter; this gate does not implement fresh generation from each updated
policy.

## Exact repeat and resume

The harness runs three serial trainer processes after the public serving stack
has exited:

1. `uninterrupted`: updates 1 and 2;
2. `split-first`: repeats update 1 in an independent process and checkpoints;
3. `resumed`: freshly loads the split checkpoint and performs update 2.

Update 1 and its checkpoint must be byte-identical between `uninterrupted` and
`split-first`; update 2 and its checkpoint must be byte-identical between
`uninterrupted` and `resumed`. The comparison also covers reference bytes and all
raw before/preclip/postclip/after/delta/moment tensors.

Checkpoints contain only adapter and AdamW tensors, CPU/CUDA RNG, schedule cursor,
ordered completed/remaining group ids, packed-input and preflight checkpoint
links, prior-update hash, optimizer-state hash, reference cache, source/model
identity, and runtime identity. Python/NumPy randomness, AMP scaling, schedulers,
partial gradient accumulation, and LoRA dropout are disabled or not used. Resume
validates all AdamW param-group settings, moment shapes/dtypes/steps, and exact
retained first-update adapter/moment bytes before loading state.

## Independent verification and tamper controls

`tests/online_grpo_optimizer_verify.py` is stdlib-only and imports no rollout,
alignment, preflight, optimizer, reward, or ML implementation. It invokes the
existing independent evidence auditor as a separate process, then independently:

- rehashes all 18 weights, model config, replay binary, and exact source files;
- recomputes every run-identity digest and binds it to preflight/rollout manifests;
- rechecks native-rendered prompt ids, raw-row log-softmax, sampled ids/masks,
  strong exports, group order, reward variance, and derived token counts;
- recomputes masked means, ratios, clipping, separate-reference KL, and losses
  using 50-digit decimal arithmetic with absolute tolerance `1e-8`;
- parses raw little-endian FP32 tensors and checks finite nonzero gradients,
  clipping, AdamW moments, bias correction, learning rate, epsilon, parameter
  updates, exact deltas, and the step-1-after/step-2-before chain;
- compares update, checkpoint, reference, sidecar, and raw tensor bytes across
  uninterrupted/repeated/resumed runs and verifies each run's `SHA256SUMS`.

Tamper controls consistently rehash affected primary records and checksums, then
require the intended semantic rejection with no report: objective value, raw
gradient, parameter delta, and resumed output checkpoint. A separate fresh
trainer control changes the resume input checkpoint's AdamW learning rate,
rehashes it, and requires rejection before any second update.

## Commands

CPU-only focused tests:

```sh
python3 -I tests/online_grpo_optimizer_test.py
python3 -I tests/online_rollout_test.py
python3 -I tests/online_grpo_preflight_test.py
python3 -I tests/online_evidence_audit_test.py
python3 -I tests/online_grpo_real_model_test.py
```

The opt-in live gate takes exact local paths and a new evidence directory. It can
stop after fresh generation/preflight with `--prepare-only`; continuing requires
that same unoptimized `PREPARED` directory and still launches separate trainer
processes.

```sh
UMMINFER_LIVE_OPT_IN=1 python3 -I tests/online_grpo_optimizer_live.py \
  --model qwen3-native --model-dir <exact-qwen3.8-directory> \
  --model-revision UNKNOWN --server-binary <isolated-gateway> \
  --stage-binary <native-qwen-stage> --replay-binary <independent-native-replay> \
  --trainer-python <compatible-sm90-python> --source-commit <40-character-commit> \
  --batch-id <unique-batch-id> --out-dir <new-evidence-directory>
```

`CUDA_VISIBLE_DEVICES=7` is enforced inside each trainer. The native stage uses
physical `--gpu 7`, the isolated gateway binds a free loopback port, and the
resident service is never contacted or restarted.

## Earlier excluded fresh-generation attempts

All attempts retain their raw records and are separate from optimizer evidence:

| Attempt | Outcome | Optimizer eligibility |
| --- | --- | --- |
| R1 | One image-only response was `not_exposed`; the actual client stopped before group export | Excluded; no optimizer |
| R2 | A prose-plus-directive response failed narrative behavior-mask coverage and was `not_exposed` | Excluded; no optimizer |
| R3 | Six covered text-only rollouts answered both simpler products correctly; both groups had zero variance | Excluded; stopped before replay/training |
| R4 | Six covered text-only rollouts: `7389×4621` was correct on all three seeds, while `4597×6839` was wrong on all three seeds | Both groups zero variance; excluded |
| R5 | `3274×8957` was wrong on all three seeds; `8642×7351` produced scores `[1,1,0]` | One group remained zero variance; whole batch excluded |
| R6 (frozen final) | `6287×5399` produced `[0,0,0]`; `8642×7351` produced `[1,1,0]` | One zero-variance group; real optimizer `NOT_RUN` |

The real Task #60 zero-variance control also exited 2 before output/CUDA with the
explicit zero-variance diagnostic. These attempts are engineering diagnostics,
not model-quality comparisons or evidence of generalized difficulty effects.

## Coverage limits

- No visual artifact or narrative directive is used. The optimizer consumes only
  text-input prompts and sampled integer-answer tokens; no visual tensor forward,
  visual/narrative optimizer execution, or visual-quality objective is claimed.
- Batched behavior, audio/video behavior, and production adapter hot-loading are
  `NOT_COVERED`/`NOT_RUN`.
- No monotonic loss decrease, quality gain, generalized improvement, or serving
  latency improvement is asserted.
- AOCI files are maintained manually from final files because the AOCI CLI is
  unavailable. No AOCI CLI execution or attestation is claimed.
