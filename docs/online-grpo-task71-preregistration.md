# Task #71 online-GRPO R1 preregistration

Date frozen before rollout: 2026-11-27.

This is the only Task #71 generation plan. Task #64 R1–R6 records, including its
partially nonzero R6 batch, remain sealed negative evidence and are not continued,
replayed as training, subdivided, or combined with this batch. No prompt, answer,
reward, seed, group order, member order, or sampling field below may be changed
after observing a Task #71 response. A failed prerequisite retains the complete
raw batch and blocks optimization; it does not authorize another plan.

## Fixed batch and exact membership

- Batch id supplied to the live harness: `online-grpo-optimizer-task71-r1`.
- Exactly two text-only groups, exactly three scheduled candidates per group,
  exactly six scheduled rollouts total, and no evaluation split (`eval_ratio=0`).
- The behavior policy is the exact local `Qwen/Qwen3.8-27B` native model loaded by
  the isolated serial public stack; upstream revision is `UNKNOWN`, and complete
  weight/config/tokenizer file hashes are bound after model load and before any
  request. The behavior-policy source revision is the committed preregistration
  execution source, not a mutable working tree.
- Group and candidate order below is normative. Dataset group ids and rollout ids
  are deterministically derived by the accepted online planner from the canonical
  plan, including this batch id; there is no replacement member or subgroup retry.

| Scheduled group order | Prompt id | Exact user message | Expected integer | Ordered seeds |
| --- | --- | --- | --- | --- |
| 0 | `online-arithmetic-9173x6842` | `Compute 9173 × 6842. Return only the base-10 integer, with no explanation, whitespace prefix, punctuation, code block, or other text.` | `62761666` | `[7101, 7102, 7103]` |
| 1 | `online-arithmetic-8642x7351` | `Compute 8642 × 7351. Return only the base-10 integer, with no explanation, whitespace prefix, punctuation, code block, or other text.` | `63527342` | `[-7201, -7202, -7203]` |

Every group has `expected_visual_modalities=[]`; no image, video, narrative, ASR,
or audio input/output is part of this gate.

## Sampling, transport, and reward

The complete sampling object is:

```json
{"max_tokens":32,"temperature":1.4,"top_p":0.98}
```

The request template adds only `behavior.logprobs=true`. Every accepted response
must be fresh, completed with `finish_reason=stop`, attested against the loaded
native identity and this exact plan, raw-SSE/CAS backed, and `COVERED` with actual
sampled token ids, clean-target masks, and full native log-softmax values.
Cancelled, truncated, malformed, stale, `not_exposed`, or transport-failed records
cannot become zero-reward candidates.

The transport object is fixed to a 600-second per-attempt timeout, at most two
attempts for a scheduled member, retry only on HTTP 503, and no transport/timeout
retry. Such bounded transport retries do not change membership or add candidates.
The split seed is `<batch-id>-train-only-split-v1`; the split is train-only.

The producer `VerifiableArithmeticReward` and independent stdlib auditor each use
only the exported record's prompt id and clean text. After removing leading and
trailing whitespace, exact decimal equality with the table's expected integer
scores 1; every other string scores 0. Unknown prompt ids fail closed. Neither
implementation may inspect serving or optimizer memory.

## Prerequisite and bounded execution

The complete batch is eligible only if all six scheduled members satisfy the
acceptance contract and **each** group has strictly positive population reward
standard deviation. The raw batch is frozen before replay/export/preflight. On
failure, all gradients, updates, checkpoints, and resume comparisons stay
`NOT_RUN`/`BLOCKED`; no offline continuation or new seeds/prompts are permitted.

On success, the accepted existing surface performs strong format-v2 token-export
alignment and independent native replay/preflight, then exactly two AdamW updates
with one complete group per update in the order above. Only the rank-1 FP32 LoRA
at `model.language_model.layers.3.self_attn.q_proj` is trainable (17,408
parameters); the BF16 base and adapter-disabled reference remain separate and
frozen. The clipped sequence surrogate uses `clip_eps=0.2`, separate-reference
Schulman KL uses `kl_coef=0.04`, log-ratio clamp is 10, learning rate is `1e-3`,
betas are `(0.9,0.999)`, epsilon is `1e-8`, weight decay is zero, and gradient norm
clip is 1. The second update intentionally reuses this one online-origin batch.

The exact gate compares uninterrupted updates 1–2 against an independent
split-after-update-1 process and a fresh resumed update 2, including raw FP32
before/preclip/postclip/after/delta/moment tensors, model/optimizer states, CPU
and CUDA RNG, completed/remaining data cursors, packed-input/source identities,
and checkpoint bytes. Independent fresh-process verification and consistently
rehashed tamper controls must pass before any optimizer result is reported.
Model quality and generalized improvement are separate evaluations and remain
`NOT_RUN` regardless of optimizer mechanics.
