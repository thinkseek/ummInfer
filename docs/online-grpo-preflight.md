# Online GRPO CPU Preflight and Run Identity

Task #50 adds `python/umminfer_online_grpo_preflight.py`, a deterministic,
stdlib-first consumer of **already accepted** online-rollout bundles. It is a
planning and identity boundary, not an optimizer. It never calls the online HTTP
executor, imports Torch/Transformers/NumPy/MLX, performs a policy/reference
forward, writes model weights, or contacts a serving process.

The accepted frozen-offline adapter `python/umminfer_joint_grpo_live.py` and the
online rollout producer remain separate. The frozen adapter still rejects online
provenance; legacy/offline dataset and pack serialization is unchanged. This
adapter accepts only online provenance and cannot resume a frozen-offline run.

## Inputs and fail-closed acceptance

```text
python3 -I python/umminfer_online_grpo_preflight.py \
  --preflight-only \
  --online-pack <accepted-online-rollout-bundle> \
  --config configs/training/online_grpo_preflight_fixture.json \
  --tokenizer-dir <exact-local-tokenizer-when-token_export-is-present> \
  --out <new-preflight-bundle>
```

The adapter first replays the source in an internal temporary directory using
`umminfer_online_rollout.replay_bundle`. This regenerates accepted trajectories,
rewards, packs, and manifests from raw SSE/CAS, and compares every accepted file
byte-for-byte. The final output directory is created only after all validation,
packing, fixture construction, and identity binding succeeds. A rejection emits
no planner result on stdout and no final output directory.

All of the following are mandatory:

1. `provenance_mode=online`, the current online plan/schedule hashes, and exact
   group and member `rollout_provenance`; absent provenance, legacy/offline rows,
   frozen-offline packs, and stale revisions fail.
2. Exactly one packed group per scheduled prompt, complete unique membership,
   deterministic reward/source order, contiguous ranks, and source-equal prompt,
   target, clean text, route, visual, and media fields. Reordered, duplicate,
   missing, extra, stale, and cross-group members fail.
3. Export-recomputed finite rewards, population-standard-deviation normalized
   advantages, exact group mean/std, and row advantages. Zero variance is valid
   and produces all-zero advantages, never NaN/Inf.
4. A completed, attested HTTP-200 raw-SSE record with `generation=fresh`, exact
   request/model/session/turn identity, validated CAS media/raw bytes, and final
   `finish_reason=stop`. Cancelled, failed, truncated, stale, mixed-request, and
   non-200 records fail.
5. `behavior_evidence=exposed` and normalized `status=COVERED` for **every**
   member. The final raw frame carries exactly one
   `umminfer-behavior-logprobs-v1` block with non-empty equal-length sampled
   `token_ids`, finite non-positive `logprobs`, and binary `mask` containing a
   supervised token. Missing values, `NOT_COVERED`, extra blocks, malformed
   values, positive/non-finite logprobs, and provenance/raw disagreement fail.
6. Exact behavior model/config/sampling identity from the accepted plan and
   attestation. Current and reference policies must declare the same model id,
   model config hash, and sampled tokenizer hash; their policy/model/weight
   revisions are separately bound and may differ. Declared current/reference
   hashes are preflight inputs, not proof that local weights were loaded.

### Independent sampled/token-export alignment

When any accepted trajectory has `token_export`, all accepted trajectories must
have it and `--tokenizer-dir` is required. The adapter invokes
`umminfer_behavior_token_alignment.validate_online_record_alignment` for every
member. This narrow record-level entry point reuses Task #48's independent
validator to check:

- exact available tokenizer files, terminal/EOS ids, and the loaded-tokenizer
  digest against sampled evidence;
- structural token_export v1/v2 ids, offsets, raw source units, spans, masks,
  source-record binding, and export hashes;
- sampled ids equal to the raw reconstructed SFT-target export ids;
- no EOS/unknown sampled supervision ids;
- clean-prose mask=1 and directive/prompt/explicit-anchor mask=0;
- masked sampled bytes exactly reconstruct NFC-normalized clean prose.

No post-hoc re-encoded clean-context ids are substituted for sampled ids. A
context-dependent clean-id difference remains a reported byte-exact relation.
The adapter performs structural alignment only; tokenizer re-encoding and
logprob numerical recomputation stay `NOT_RUN`. With no supervision exports,
alignment is honestly `NOT_COVERED_NO_EXPORTS`; sampled behavior evidence is
still mandatory.

## Exact current, behavior, and reference inputs

`configs/training/online_grpo_preflight_fixture.json` demonstrates the strict
config schema. `current_policy` and `reference_policy` each contain:

```text
policy_id, revision, model_id, model_revision, model_sha256,
model_config_sha256, tokenizer_sha256
```

`prompt_token_ids` maps every group's exact `prompt_sha256` to an explicit,
bounded, non-negative id list. This avoids re-tokenizing sampled responses and
makes prompt bytes/ids an auditable preflight input. The committed ids are
contract-fixture values, not real tokenizer output. A real gate must independently
re-derive them from the exact loaded tokenizer and prompt before execution.

For each member, with prompt ids `P`, sampled response ids `R`, and sampled
response mask `M`:

```text
input_ids     = P + R
labels        = [-100] * len(P) + [R[t] if M[t] == 1 else -100]
response_mask = [0] * len(P) + M
```

The causal contract is `model_logits[t-1] -> labels[t]`. Prompt positions and
unsupervised directive/prompt/anchor positions never contribute to the response
mean. Behavior logprobs remain target-length arrays; no fabricated prompt zeros
are inserted. The bundle binds hashes of prompt ids, sampled response ids, packed
ids, labels, masks, their ordered combination, current-policy input identity, and
reference-policy input identity.

## Objective fixtures — arithmetic only

The reduction matches the accepted Stage C sequence-level form, but online old
and reference policies are distinct. For supervised response positions `S`:

```text
mu_current   = mean(current_logprobs[t]   for t in S)
mu_behavior  = mean(sampled_logprobs[t]   for t in S)
mu_reference = mean(reference_logprobs[t] for t in S)
d_ratio = clamp(mu_current - mu_behavior, -C, C)
ratio = exp(d_ratio)
clipped = clamp(ratio, 1-epsilon, 1+epsilon)
surrogate = min(ratio * advantage, clipped * advantage)
d_kl = clamp(mu_reference - mu_current, -C, C)
KL = exp(d_kl) - d_kl - 1
loss = -surrogate + kl_coef * KL
group_loss = arithmetic_mean(member losses)
```

Unlike frozen offline GRPO, behavior is not silently reused as reference.
Current/reference logprobs in `objective_fixtures.json` are generated by the
configured cyclic offsets solely as deterministic CPU arithmetic fixtures. Every
member labels them `SYNTHETIC_CPU_FIXTURE_NOT_MODEL_OUTPUT`; both forwards and
new numerical logprob recomputation are `NOT_RUN`.

Tests pin positive-advantage clipping at ratio 2.0 to surrogate 1.2,
negative-advantage min-selection at -2.0, zero KL when current equals reference,
and Schulman KL `exp(-1)` for reference-minus-current `-1`. Multilingual,
NFC/emoji, visual, explicit-anchor, and zero-variance groups exercise the full
packing path in both token_export v1 and v2.

## Run, split, and checkpoint identity

`run_identity.json` hashes the canonical identity into `run_id`. It binds:

- canonical plan bytes/hash and independent rollout schedule hash;
- rollout and pack manifest hashes and every accepted source file hash;
- adapter/online/replay/alignment/token-export/dataset/reward source hashes;
- resolved config, behavior/current/reference policy and tokenizer identities;
- every group export, member source export, raw SSE, behavior evidence, packed
  input, current input, and reference input hash in deterministic order;
- exact objective, advantage, zero-variance, and alignment semantics.

`schedule.json` deterministically splits plan-order groups using
`groups_per_step`; changing that split changes the config and run identity. Every
step binds its group/member/packed hashes and the checkpoint identities before
and after it. Checkpoint descriptors exist for every cursor and bind run id,
next step, schedule position, completed/remaining group ids and export hashes,
packed order, and all three policy hashes. They contain no fake tensor or
optimizer state: `optimizer_state=NOT_RUN` and `model_weights_updated=false`.
Resuming requires exact run/checkpoint/group/member/packed/policy identity, so
online/offline runs and different behavior, current, reference, or split
revisions cannot cross-resume.

## Independent fresh-process verification

```text
python3 -I python/umminfer_online_grpo_preflight.py \
  --verify-bundle <existing-preflight-bundle> \
  --online-pack <same-accepted-online-rollout-bundle> \
  --tokenizer-dir <same-exact-tokenizer-when-required> \
  --out <new-reconstructed-bundle>
```

The verifier reads the config bound inside the preflight bundle (no replacement
`--config`), verifies `SHA256SUMS` and the manifest sidecar, replays and rebuilds
from source, and requires the exact file set and byte-identical contents before
writing the reconstruction. Both stale-checksum tamper and consistently rehashed
objective tamper fail with no output. Tests invoke this command in a fresh
`python3 -I` process with ML imports blocked.

## Deterministic evidence and exact hashes

Machine-readable evidence is in
[evidence/online-grpo-preflight-cpu.json](evidence/online-grpo-preflight-cpu.json).
The v1 three-group/nine-member fixture record includes:

| Object | SHA-256 / value |
|---|---|
| Canonical plan identity | `e3d1629fa9b334620886c5dbfd6c4c2cb402de6e88ba94974f24419572ba46e5` |
| Canonical plan file | `d7f6cf79090783e0783e155eac3d56002f0c8f983d7ef605464488a8ff969402` |
| Rollout schedule | `8697e83da9be41b56b5bb5985a627c4441dd53d3c866d492ee97c6c20c408bcc` |
| Config canonical object | `47ac0f75ada2ec92da7a322a1453903f6ae32ac5ce4c4b91ec44ae933a96ab91` |
| Source trajectories | `d84554a3f22c915d0f1ee9ba08b7afc86d397e4133d50c07d5eb263ee6a897e7` |
| Source pack manifest | `467e92a541115c87ae4e8d953de86c21318a9f4a1aa0bb7f8999ce80bf98650f` |
| Accepted source file-set identity | `de31842ba422edea7662dc29572a1d081fef39b93d29a5f727585c783a36ae31` |
| Online preflight run id | `43d4bd01c7ed6736313af3aa71aae638bef9964c79e6ff5fbdba245371a84a21` |
| `run_identity.json` | `7e9e229308c2b7ee22e467b7c762073d76b4a44291a1fd3abef2ef058eb55277` |
| `schedule.json` | `5673567fc9bd75010e20ef430098964cfef40df8ae15207d0d1a2cf2ebbc6dd1` |
| `objective_fixtures.json` | `337b69484cd9b4473e7255fcb0e69c40516d6482d9b62f6175140e54a2adffe2` |
| `manifest.json` | `7f67d81b0232ac1dbd002d1e7f7800a96d7d04a9342400e2ed2e7a3985d386cd` |
| `SHA256SUMS` | `276b21f4523a3ae4a6b83a1a6be1f67bf4b0a8ed892fa6ddcef36f29409a23a4` |

Fresh-process reconstruction: seven files, `BYTE_IDENTICAL`, Torch not imported.
The zero-variance group has mean `1.0`, std `0.0`, three zero advantages, and a
finite KL-only fixture loss `7.273936252207847e-06`. The complete per-group
losses, policy identities, source-code hashes, and claims are in the evidence
JSON. These are deterministic contract hashes, not model-output or training
quality evidence.

## Tests and explicit boundaries

```text
python3 -I tests/online_grpo_preflight_test.py
python3 -I tests/online_rollout_test.py
python3 -I tests/behavior_token_alignment_test.py
python3 -I tests/joint_grpo_live_test.py
umm-test-local
```

The first four CPU suites passed 5/5, 11/11, 8 plus one real-tokenizer skip, and
4/4 respectively. Related CPU suites also passed: token export 9 plus one
real-tokenizer skip, raw-SSE replay 9/9, dataset 8/8, planner 6/6, SFT 6/6,
DPO 10/10, and RL architecture 16/16. `umm-test-local` passed, including its C++
contract, realtime, voice, and online-attestation coverage. Real tokenizer strong
alignment remains `NOT_RUN` unless its explicit local gate is supplied. No GPU
was used.

A future **real online-generation + optimizer** acceptance gate must additionally:

1. Execute a fresh public online plan against a trusted loaded native endpoint;
   independently re-hash model config, all weights, tokenizer, binary, and source;
   bind every actual response and retry/cancel outcome to fresh raw SSE/CAS.
2. Re-encode supervision with the exact tokenizer and independently recompute all
   sampled logprobs in a fresh forward from exact prompt/sampled prefixes. Keep
   context-boundary token differences explicit; never repair sampled ids.
3. Fingerprint actual current/reference local model files and derive exact prompt
   ids. Recompute current and reference logprobs from the packed inputs and causal
   shift, replacing every synthetic fixture value before claiming an objective.
4. Compare real loss/gradient/update values against independently derived fixtures
   under stated dtype/tolerance rules; cover positive/negative advantages,
   clipping, zero variance, masks, and separate behavior/reference policies.
5. Save and verify real model/optimizer/RNG/schedule/checkpoint state in a fresh
   process, including interrupted split/resume and rejection of every changed
   behavior/current/reference/group/member/packed identity.
6. Keep real online generation, online optimizer execution, GPU use, serving,
   batch/audio/video behavior evidence, model quality, generalized improvement,
   and production training `NOT_RUN`/`NOT_COVERED` until their own measured gates
   pass. This CPU preflight does not advance any of those claims.
