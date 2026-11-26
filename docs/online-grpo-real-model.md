# Real Qwen3.8 online-GRPO preflight and sampled/logprob alignment

Date: 2026-11-27. Initial execution base: official main
`49d34063b7ecb61d0571bf962b11f1a71c0d039f`; final integration base:
`27c584e` (Wave 86).
Execution commits:
`b07dd8d96235d2c4cd0b56c60fa8003562c65eb4`,
`4813186ddc5b2d79ee0824cb702d878c749fc5d4`, and
`a06ba87014d1e850ef3cffb2e171345d7df2ba35`.
The corresponding Wave 86 integration chain is
`3d00caf`, `6311325`, `7e4dd67`, `4108769`, `fd282ae`, `824e1dd`, and
`a6a3b51`. The final live gate executed
`1894a66c7513b69d751d74d4f049808354d7153d`, which also included
`b2f6931d5a7ffe235d7a9b9579f55349cb5cbf60`,
`f84bdaea5f50f00aa33241b6b3181f65b3eec8a6`, and
`37b4f465a65d3d1a8369fd70e32820a3ca89ac69`.

This is a real-model preflight, sampled-behavior/token alignment, and numerical
objective-validation result. It is **not** an online optimizer update, an
exact-resume/update result, a training-quality result, or a generalized
improvement claim. Those remain `NOT_RUN`.

## Exact model and retained exposed rollouts

The gate reuses only the accepted Task #49 attempt-6 exposed rollouts after
revalidating the complete retained source summary, plan, schedule, raw-SSE/CAS
records, producer binaries, loaded native process identity, and exact local
model. No tiny or substitute model is used. The upstream model revision remains
`UNKNOWN`; the exact files, rather than a revision label, define the compared
model.

| Identity | SHA-256 / value |
| --- | --- |
| Model | `Qwen/Qwen3.8-27B`, native `qwen3-native`, `Qwen3_5ForConditionalGeneration` / `qwen3_5` |
| Canonical 18-shard weight manifest | `e38893095032f8fb1115c76a2e264acb8a13c5233ee69ad3fd73d46c27386613` |
| `config.json` | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| Loaded three-file native tokenizer manifest | `1dd1199e35f6cfb9f547458e02c9fb3a522e9c346ed22ba3d461aeed14310016` |
| Complete 29-file token-export available-files manifest (18 weights + 11 config/tokenizer files) | `2a9102aa1f79fcef15ed4694bea515d05fe1efbbf3eb234af46013ff84b5b881` |
| Independent native replay binary | `c4ca6a3ed4721a9c615572ed928a108344cf3c4fc3a244a43ad62891a0852b47` |
| Retained source trajectory | `d7b114285d54271c24da39180584c59d19476322f0142cc10be2f801ef9b05ad` |
| Retained source pack manifest | `501e3a2f8386bacdd4e97852e5ad04de876030995ebf7b4ef8d35da553aaf6e8` |
| Retained semantic plan | `b461af977ff04ed7e156d1fde5b94233dfd990479095c96fdd8442fd7282b8da` |
| Retained schedule | `45188f2d4970d9f0f219e83b982b84a2d3a2ab947fef08dd697f14c8e1c4ce46` |
| Retained behavior policy | `1f45a7ec17a880338c3713264b5ea9d3b83966b0f45d575a5e6280022b49b513` |

The two complete groups contain three unique rollout/request/session identities
each: ASR-free multilingual text and auto-narrative interleaved image text.
The PNG remains a CPU contract fixture, not a generated-image-quality result.
The retained default-off, truncated, cancelled, unsupported, and real-retry
negative evidence is unchanged and remains outside accepted groups.

## Additive token-export and real preflight path

`tests/online_grpo_real_model_live.py` attaches format-v2 token exports with the
exact local fast tokenizer (`transformers 5.6.0`, `tokenizers 0.22.2`). The fresh
export process strongly re-encodes and validates every record. The gate then:

1. independently revalidates every accepted record against the original plan,
   schedule, response attestation, behavior-policy digest, and raw-SSE/CAS bytes;
2. requires sampled ids to equal raw reconstructed SFT-target ids, checks v2
   normalized/original source units and source spans, excludes EOS/unknown
   tokens, and requires clean prose mask=1 with directive/prompt/anchor mask=0;
3. preserves context-boundary clean-tokenization differences explicitly and
   never substitutes post-hoc clean-context ids for sampled behavior ids;
4. rebuilds the complete one-to-one dataset `group_id` membership, ordered
   rewards/advantages, and pack manifests, then requires byte-identical
   fresh-process replay;
5. derives exact rendered native prompt ids, constructs
   `input_ids = prompt + sampled_response`, causal labels, and response masks,
   and runs the existing `--preflight-only` adapter plus fresh reconstruction;
6. runs the Task #52 independent auditor again and requires zero checkpoint
   disagreements. Its sealed 16 pre-fix disagreements and the dataset-`group_id`
   fix are not modified or normalized away.

The existing preflight document's `objective_fixtures.json` remains explicitly
synthetic arithmetic evidence. The new `real_objectives.json` is a separate
schema and evidence class containing actual native current/reference outputs;
the synthetic and real run/objective identities are never aliased.

## Independent raw-logit and objective verification

The serial native engine has a validation-only
`pre_sampling_logits_observer`. When enabled it copies the same FP32 LM-head row
used for token selection but does not enable or call the producer's
`selected_token_logprob`. The callback runs only for emitted non-EOS tokens.
With no observer or producer capture, the historical greedy no-logits-copy fast
path is unchanged. Batched evidence remains `NOT_COVERED`.

`native/qwen38_tests/independent_logsoftmax.hpp` deliberately does not include
producer sampling/scoring code. It recomputes the selected score using
extended-precision max, sum, and logarithm over the FP32 row. Every emitted token
retains its id, mask, raw-row SHA-256, selected raw logit, row maximum, log
normalizer, and resulting log-probability.

The schedule is strictly serial on physical GPU 7: for each of six scheduled
rollouts, run cold native behavior, current, and reference processes in that
order. All 18 loads use the same 18-shard weights, config, tokenizer, and replay
binary. Current/reference terms are neither dropped nor replaced by behavior,
and no different-weight comparison is allowed. Each process records its pid,
start-time ticks, executable, return code, document hash, and GPU-before/after
state.

`tests/online_grpo_real_model_verify.py` is a separate stdlib-only fresh-process
verifier. It imports no rollout, alignment, preflight, producer objective, or ML
module. It independently rehashes the model/tokenizer/binary, re-parses final
raw-SSE behavior blocks, checks EOS exclusion and one-to-one group membership,
reconstructs causal packing and raw-row arithmetic, and uses 50-digit decimal
means/exponentials to recompute:

```text
d_ratio = clamp(mean(current) - mean(sampled_behavior), -C, C)
ratio = exp(d_ratio)
surrogate = min(ratio * advantage,
                clamp(ratio, 1-epsilon, 1+epsilon) * advantage)
d_kl = clamp(mean(reference) - mean(current), -C, C)
KL = exp(d_kl) - d_kl - 1
loss = -surrogate + kl_coef * KL
```

The absolute per-token tolerance is `1e-9`: producer scoring uses FP64
accumulation while the independent native path uses extended precision over the
same FP32 row, so only floating-point reduction rounding is allowed. The
absolute objective tolerance is `1e-12`: producer evidence uses FP64 formulas
while the independent verifier uses 50-digit decimal arithmetic. Any larger
per-token or objective disagreement fails without a report. Separate arithmetic
controls cover positive-advantage clipping, negative-advantage min selection,
log-ratio clamping, and the Schulman `exp(-1)` oracle; they are not presented as
model outputs.

## Commands

The deterministic CPU suite is:

```sh
python3 -I tests/online_grpo_real_model_test.py
```

The opt-in real gate takes an exact local Qwen3.8 model directory, the accepted
Task #49 bundle and summary, an independently built native replay binary, an
exact source commit, and a new evidence directory:

```sh
UMMINFER_LIVE_OPT_IN=1 python3 -I tests/online_grpo_real_model_live.py \
  --model qwen3-native --model-dir <exact-qwen3.8-directory> \
  --model-revision UNKNOWN --replay-binary <independent-native-replay> \
  --source-rollout-bundle <accepted-task49-bundle> \
  --source-live-summary <accepted-task49-summary> \
  --source-commit <exact-40-character-commit> --out-dir <new-evidence-directory>
```

The standalone independent verifier is:

```sh
python3 -I tests/online_grpo_real_model_verify.py <evidence-directory> \
  --model-dir <same-exact-qwen3.8-directory> \
  --replay-binary <same-independent-native-replay> \
  --report <new-report.json>
```

## Regression evidence

The following checks passed on the Task #60 implementation before and after the
Wave 84 rebase; they do not substitute for the real-model gate:

- `umm-test-local` in Debug and ReleaseFast, and again on `ac25f5c`: exit 0,
  including native, unified, realtime, voice, and online-attestation coverage;
- new CPU validation: 6/6;
- online rollout: 13/13; online preflight: 6/6; independent online audit: 12/12;
- behavior-token alignment: 9 tests `OK` with the one optional real-tokenizer
  case skipped only in the no-local-tokenizer CPU invocation; token export: 10
  tests `OK` with the equivalent optional skip; behavior harness: `PASS`;
- raw-SSE replay: 9/9; standalone online attestation with the installed
  ReleaseFast gateway: 4/4;
- Stage C pipeline: 16/16; frozen GRPO: 4/4; RL architecture: 16/16; dataset:
  8/8; joint planner: 6/6; joint SFT: 6/6; joint DPO: 10/10;
- multimodal deterministic replay: 70/70 cases with matching replay metrics;
- Web UI contract on Wave 84, including keyboard shortcuts: `PASS`;
- SM90 CUDA 12.4 native replay build and `qwen38_host_tests`: `PASS`.

The joint SFT/DPO/GRPO default invocations again printed their expected real
optimizer `NOT_RUN` boundaries. The optional CPU tokenizer skips are not counted
as real-tokenizer results; the separate live gate performs the exact local
strong export. A standalone attestation invocation before installing the local
gateway failed with file-not-found errors and was rerun successfully with the
installed binary; it is not counted as a code failure or as an additional pass.

## Measured result, tamper controls, checksums, and limits

The final measured counts, per-token/objective deltas, tamper outcomes, evidence
hashes, test totals, and cleanup state are recorded below from the terminal
bundle. No optimizer was started and no training-quality gain is claimed.

### Final R6 result

| Check | Measured result |
| --- | --- |
| Status | `PASS` at execution commit `1894a66c7513b69d751d74d4f049808354d7153d` |
| Groups / rollouts | 2 complete dataset groups, 3 unique members each; 6 total |
| Sampled / supervised / directive-zero tokens | **192 / 156 / 36** |
| Strong token export | v2, 6 records, 29-file available-files manifest; R3 fresh export retained and independently revalidated |
| Accepted replay / preflight | 17 files byte-identical; seven preflight files byte-identical; fresh independent audit `PASS` |
| Real preflight run id | `3db56b3523e8dfe2d7cb248db79772b1323a4757ac92426a067af2476ab1b24f` |
| Cold native schedule | 18 unique pid/start-time identities, serial behavior→current→reference per rollout |
| Behavior independent vs sampled max abs per-token delta | **`2.580384863271764e-12`** (tolerance `1e-9`) |
| Current vs independent behavior max abs per-token delta | **`0.0`** |
| Reference vs independent behavior max abs per-token delta | **`0.0`** |
| Raw FP32 rows | Exact row hashes/components across behavior/current/reference for every token |
| Maximum independent objective delta | **`7.946142857142661e-17`** (tolerance `1e-12`) |
| Real group losses | `[0.0, 0.0]`; both groups have zero reward variance and zero advantages |
| Raw SSE / group / packing / EOS checks | 6 raw streams reparsed; one-to-one membership and causal packing exact; EOS excluded |
| Model files rehashed by independent verifier | 18 weight shards plus the 29-file available-files manifest and replay binary |
| Tamper controls | `objective-value`, `current-logprob`, `directive-mask`, `group-membership`: all exit 2 with **no report** |
| GPU cleanup | GPU 7 `3 MiB`, 0% utilization, no compute process before and after; harness exited |

The zero real losses follow from the frozen groups' zero reward variance and
identical policy/reference weights; they are not a training-quality result.
Nonzero advantages, clipping, negative-advantage min selection, clamping, and
Schulman KL are exercised separately by the clearly labelled decimal arithmetic
controls.

### Evidence hashes

The compact summary is
[online-grpo-real-model-sm90.json](evidence/online-grpo-real-model-sm90.json);
raw per-token and per-group objectives are in
[online-grpo-real-model-objectives-sm90.json](evidence/online-grpo-real-model-objectives-sm90.json),
and the independent decimal report is in
[online-grpo-real-model-independent-sm90.json](evidence/online-grpo-real-model-independent-sm90.json).
The complete raw evidence bundle also retains accepted raw SSE/CAS, all source
spans, all 18 native stdout/stderr/document triples, process identities,
replays, preflight reconstruction, logs, and tamper diagnostics.

| Object | SHA-256 |
| --- | --- |
| Compact committed summary | `7e3e287ad220e1bcbbe736af8de71524d8275c55e9da9b3b586b9d7408ae18ca` |
| Raw per-token/group objectives | `ce7137271c8033f2638280205a2664b39dc2f3e8dc4f1fdaba52cd246999cd7f` |
| Independent verification | `440a6579ca83b3389f9df1ec0bdc324b7f1aa3c0fc82057fd8adec6996175137` |
| Independent structural/preflight audit | `9c1ca8e98d45efe1a00f84ca8a069838527c710f473f0c7fdc288fc8fe99cb69` |
| Tamper controls | `44dfc0fc3dbd3e317e889739cbf35d4583fd5bed61e7a38bd696e789cbbe21de` |
| Derivative accepted trajectories | `4374cfac205cc0fe9a74c998607a15396a2e70f352b3283466a1d37d6a73212f` |
| Derivative rollout manifest | `cc6e5c4ce3d152094e537cdb0e11e3470d356682f93792811157fd998f5c0ae7` |
| Derivative pack manifest | `931e0cf89b2d258c3f3edd2ee63a6476f4d4f21f7b5f45f50ff6e17f9eb8613f` |
| Preflight manifest | `36cd62f94b67cf0e5298a7010b8de559ea17e5de12c3c573ed1cac2903bb2d01` |
| Preflight `SHA256SUMS` | `04df7505b7b283adc0c3160b5bd1510cba3e46cb3a87131406399bba3922dfeb` |
| All 18 forward-process records | `4e9bdd5ecac7b282900b9d763db19048f35352ca8ea252a7da25a5f36f3f57f2` |
| Raw live `summary.json` | `59c0c3e91622c673aa883a757ef678cfb186de62821a6da6b4c782fc0d5a6b80` |
| Raw 112-file `SHA256SUMS` | `2a4ec883da0addcc8c7239828a7cd4cc50fd95f29cbf079dd024a43c31d7bbfa` |

`sha256sum -c SHA256SUMS` verified all **112/112** raw evidence files from the
bundle root. A separate recheck initially ran from the wrong working directory
and reported missing relative paths; rerunning from the evidence root passed.
This is not counted as an evidence failure or an additional gate.

### Earlier failed attempts retained

All five earlier attempts stopped **before any GPU forward**, left GPU 7 at
`3 MiB`, and remain separate from the accepted R6 bundle:

| Attempt | Exact failure | `summary.json` SHA-256 |
| --- | --- | --- |
| R1 | Real export exceeded the initial 300 s CPU timeout | `642b44c82c1cf6991c3b7887e58e4898efa8b5fd595f245b664503b842c51bad` |
| R2 | Harness read nonexistent `sampled_ids_replaced` after strong export | `c2f55a5c4d578498e28a28d5233329188b554536554a0ff7407370d317f57653` |
| R3 | Independent auditor hard-coded attempt 0 and rejected accepted attempt 2 | `f1714d8085a0f42d0a2387429b7d427e342993d6a98d59a2e08d1004970747fe` |
| R4 | Generic 64 MiB read cap rejected a real weight shard | `7adf40753448c6eebd75e982e3a8d80b87816c8bfbfb408a916625001fcf5d7d` |
| R5 | Independent EOS check missed `added_tokens_decoder` key fallback | `a45395f31bc768309e4bbd0372d88870018e6bbb9fbff0891a6b9360050960ca` |

No assertion was weakened to convert those failures into success. R3's accepted
export/preflight was reused only after exact identity, replay, reconstruction,
and checksum checks; R6 then ran a fresh corrected independent audit and all
model/numerical gates.

### Explicit limits

- **Optimizer and exact resume/update: `NOT_RUN`.** No real optimizer state,
  gradient, model update, or interrupted optimizer resume was executed. The
  preflight checkpoints remain identity descriptors only.
- **Model quality and generalized improvement: `NOT_RUN`.** No training-quality
  gain, deployment improvement, or generalized performance claim is made.
- **New online generation: `NOT_RUN`.** The accepted Task #49 exposed rollouts
  were reused after exact identity checks; no new public serving capture is
  claimed. New native current/behavior/reference forwards did run.
- **Coverage:** batched behavior and audio/video behavior remain `NOT_COVERED`;
  the interleaved PNG remains a CPU contract fixture.
- **Model revision:** upstream revision remains `UNKNOWN`; complete file hashes
  define the exact compared model. All current/reference/behavior passes used
  those same weights.
- **AOCI:** `aoci.code.txt` and `.aoci/baseline.json` are maintained manually
  from final files because the AOCI CLI is unavailable. No AOCI CLI execution or
  attestation is claimed.
