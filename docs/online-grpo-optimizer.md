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

This document preserves the Task #64 blocked gate as historical evidence and adds
the Task #71 successful bounded execution. Neither result claims a model-quality
gain, generalized training improvement, production deployment, or a new native
serving capability.

## Task #71 measured outcome — real bounded optimizer and exact resume `PASS`

Task #71 used the existing Task #64 surface on official Wave 93 main
`413cc9b4114ac83696925aeb43577e39853977a8`, satisfying the required Wave 92
minimum `869160f3b56aabce15cb792963d85f425965af89`. The frozen plan was committed
before rollout as `5fa9e4960596fe56d3f12b1ee8b8e9726bfc5bf8` and rebased without
content change as `7c9e6d7ef9b9478dc006c16be15d5e1a9512bc59`. The final optimizer
execution source is `58feb7d5c0542c6ba0ad0196c50ef3bee222140b`, with run id
`398b867627358015c903e8e42a5368bc76c21d95b358eeafea5bc7ce6cc99843`.
The preregistration is [online-grpo-task71-preregistration](online-grpo-task71-preregistration.md);
the path-free compact evidence is
[online-grpo-optimizer-task71-sm90.json](evidence/online-grpo-optimizer-task71-sm90.json),
SHA-256 `defda7dadfb79e2804a435b51e290acb885c59dd3c410e17d1fdc509d94a0702`.

### One frozen valid batch

No Task #64 record was continued, no Task #71 prompt/reward/seed/member changed
after observation, and no new rollout was generated for either implementation
fix. The only Task #71 batch was `online-grpo-optimizer-task71-r1`, with two
complete text-only groups of three, temperature 1.4, top-p 0.98, and at most 32
new tokens. Both groups produced ordered rewards `[1,1,0]`, population standard
deviation `0.4714045207910317`, and advantages
`[0.7071067811865476,0.7071067811865476,-1.414213562373095]`.

| Order | Prompt id | Dataset group id | Exact clean answers |
| --- | --- | --- | --- |
| 0 | `online-arithmetic-9173x6842` | `7f03cf32b15052f3ae29479cd0db32b9` | `62761666`, `62761666`, `62761766` |
| 1 | `online-arithmetic-8642x7351` | `f22f11813e1e5b0546f3ab9ab2b3a2b8` | `63527342`, `63527342`, `635` |

All six fresh public responses were attested, completed, raw-SSE/CAS backed, and
`COVERED`. There were 43 sampled and 43 supervised tokens, with no directive
masks. Frozen identities include plan-file SHA-256
`bb03c8d05137ad9cd79e86ff809719c2ea3a1d8a5c20777e43e4473f131af5bc`, semantic
plan SHA-256 `477a9ddaf57e5df32c3d3385e5f6c979930ff35d2d04283584db91a647fa2835`,
and schedule SHA-256
`1e51cf73059f2a37a60823724ec344469b9595d796410ebbffb428c8ba4470b2`.
The preregistration document itself is SHA-256
`a60e427ccc8ae83b676e17742d4c93fb6c6f09826506b9578a9d7271cdcaa145`.

### Strict export, replay, and preflight

The exact Qwen3.8 fast tokenizer produced six fresh format-v2 exports with strong
re-encoding. Six independent cold native replays reproduced every sampled id and
mask; maximum sampled-logprob difference was `6.667583152264456e-14`, within the
`1e-9` tolerance. Fresh online replay passed byte-identically with 16 accepted
files. Accepted trajectory and pack-manifest SHA-256 values are
`b0e59686efd9130b1e2cc4d4f82ac15c61d948790f5002a45dc6814033826b66` and
`c0411a08375484943c11778ce0d78270f8819602b0237907a2fc1118564969a2`.
The preflight run id is
`8e429a7373bb6548349f66d4cad52a52af99b0a71a83e0821e396575a4c4bb54`; its manifest
SHA-256 is `48fd11819516286ddf298aa74a5a31dbc1758cbd4dc290c9ecb3ad64ae1974d9`.
Seven preflight files reconstructed byte-identically, and the independent audit
passed with zero checkpoint disagreements.

### Real gradients, AdamW, and parameter deltas

Only the rank-1 FP32 LoRA at
`model.language_model.layers.3.self_attn.q_proj` changed; the 18-shard BF16 base
and adapter-disabled reference remained frozen. The two whole-group updates used
the accepted clipped sequence surrogate and separate-reference Schulman KL.
Initial current/reference values were exactly equal; step 2 had nonzero KL after
step 1. Step 1's zero LoRA B explains why only B's 12,288 elements changed; step 2
changed both A and B.

| Step | Loss | Raw gradient norm | Nonzero gradients | Changed scalars | Max abs delta | Delta L2 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `0.0020096665326077754` | `0.0017040768871083856` | 12,288 | 12,288 | `0.0009999864269047976` | `0.10977293453169103` |
| 2 | `-0.0027834560738428937` | `0.002331538824364543` | 17,408 | 17,408 | `0.001000919728539884` | `0.09461829327523935` |

Update 1/2 SHA-256 values are
`9a51da64f27d228f2916702c4c5b51b54a1511df075eb3c0d4c0c7b2dde9a21e` and
`91e43f09370c4b4e11bc503061dd47a38cba4bd66fa28095099511c75e8ca913`.
Checkpoint 1/2 SHA-256 values are
`d1f62a0b3f5e29d8653d130fc00c6d2a1e0b7ec53b75129edc0cc0aea22d0487` and
`268ae691aecf02d81797a3f6076269a4add337abf6f955b1e1c443016aea43ca`.
The shared frozen reference document SHA-256 is
`d90362da02973e16e5febd69b8d3566c2220a80133ca9923ae98407c854ce0d2`.
The update loop took `12615.018 ms` in the first execution; this is not an
end-to-end serving latency or a performance-quality comparison.

### Exact uninterrupted versus resumed execution

Three serial fresh trainer processes ran uninterrupted updates 1–2, split update
1, and resumed update 2. Update documents, adapter/AdamW checkpoints, sidecars,
reference bytes, and every raw FP32 before/preclip/postclip/after/delta/moment
tensor were byte-identical at both comparisons. The checkpoint bytes include the
CPU/CUDA RNG states, schedule position, completed/remaining groups, packed-input
identity, model/source identity, and prior-update/optimizer-state links. The
resumed process reported initial step 1 and exactly one executed update; the other
cursors were 0→2 and 0→1 respectively.

The stdlib-only independent verifier imported no producer scoring, preflight,
alignment, optimizer, reward, Torch, Transformers, or NumPy module. Its separate
auditor reconstructed raw-SSE/CAS/export/preflight identities, and its decimal
objective maximum absolute difference was `4.216757951492635e-16`. Independent
AdamW maximum absolute recomputation errors were `2.3283064365386963e-10` at step
1 and `9.313225746154785e-10` at step 2, within the verifier's established FP32
tolerances. Both initial/final independent reports are byte-identical with SHA-256
`d2a92295e080d5b382043711366aaf1dc4b0f87f818e22c550cbf3d8b5c08393`.

Consistently rehashed tampering of objective values, raw gradients, parameter
deltas, and resumed checkpoint bytes each exited 2 with no report. A fresh trainer
also rejected a structurally rehashed checkpoint with altered AdamW learning rate
before update 2 (`param-group/hyperparameter mismatch`). Group membership, source
ids, masks, cursors, nonzero-gradient claims, and nonzero-delta claims are checked
against the independently audited preflight and raw tensors, not producer claims.

### Real-evidence fixes and retained failures

Two focused issues were found and fixed without changing the frozen batch:

1. The original fresh resume wrapped restored model and moment value rows in an
   extra list, causing `struct.error: required argument is not a float` before
   update 2. A deterministic tensor-row regression failed pre-fix (10 tests, one
   error) and passed post-fix (10/10). Commit
   `66bcefb1c9ef9be8982cd13943b92f8b039f86ba` fixes only those two resume byte
   checks. All 14 raw uninterrupted tensors remained byte-identical before/after.
2. The first fixed run's gradient tamper was correctly rejected by the independent
   verifier as `gradient norm mismatch`, but the harness accepted only two other
   diagnostics and therefore failed its own control. A matcher regression failed
   pre-fix (11 tests, one error) and passed post-fix (11/11). Commit
   `58feb7d5c0542c6ba0ad0196c50ef3bee222140b` adds that exact diagnostic to the
   harness matcher; no verifier assertion or tolerance was weakened.

The original failed summary SHA-256 is
`374b49988afd5eafe6447d9bfab31df8fa49c7a7a8cd642abba8da8eb912a644`; the
intermediate exact-resume-pass/tamper-harness-failed summary is
`eab31675145961b10e7f984eac0c636966360e214fa68faded5f2c49d22af6de`.
Both remain linked in the final compact evidence. No failure was normalized into
training data, and each final trainer process was rerun under the final source and
run identity using the same prepared batch.

### Final checks, evidence integrity, and limits

Final local checks passed on the Wave 93-integrated fixed source:
`umm-test-local` Debug and ReleaseFast; optimizer 11/11; online rollout 13/13;
online preflight 6/6; independent online audit 12/12; real-model CPU 6/6; raw-SSE
replay 10/10; Stage C 16/16; joint trainer 6/6; RL architecture 16/16; standalone
online attestation 4/4; and `tools/check_feature_map.py`. Token-export and sampled
alignment tests pass with only their optional no-local-tokenizer fixture skipped;
the live gate performed the actual strong Qwen validation. An extra standalone
attestation invocation before installing the local gateway failed with
file-not-found and passed after the build; this was not a contract failure.

The final raw evidence bundle has 157 checksummed files, raw summary SHA-256
`e52e49ed6cf42ed5d04282936689d79cb49f06a4f366fa8f1cd529ba54b2abd7`, and
`SHA256SUMS` SHA-256
`3ca38d6b90a8c131a05e77cb255a380698aeda436bcdd180a6cc4790f3b95a07`.
GPU 7 returned to `3 MiB`, 0% utilization, and no compute or task-owned process;
the resident service was never contacted or restarted. Model quality, generalized
improvement, production training, and quality gains remain `NOT_RUN`/unclaimed.
AOCI files are hand-maintained because the AOCI CLI is unavailable; no AOCI CLI
execution or attestation is claimed.

## Historical Task #64 measured outcome — real optimizer `NOT_RUN`

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
