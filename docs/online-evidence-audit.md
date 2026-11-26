# Independent online-rollout, sampled-alignment, and online-GRPO evidence audit

Date: 2026-11-26. Base: `2e1283c8f2748277def22d317ca8e13768540564`
(Wave 64). This task is CPU-only and does not change serving, rollout, alignment,
or dataset-scoring implementations. The only runtime change is the two-line
checkpoint group-id identity fix recorded below (`ea9155d`).

## Verdicts at a glance

| Evidence | Independent result | Meaning |
| --- | --- | --- |
| Retained real Task #42 text/visual captures | **PASS — STRUCTURAL** | Exact raw-SSE/CAS/request/attestation/token-export/source-span/mask reconstruction; no new tokenizer or numerical forward |
| Deterministic v1/v2 online-rollout sources and packs | **PASS — BYTE_IDENTICAL** | Plans, schedules, requests, policy/sampling identities, complete groups, rewards, DPO/GRPO rows, manifests, and accepted file sets independently rebuilt |
| Online-GRPO run identity and objective fixtures | **PASS — BYTE_IDENTICAL** | Current/behavior/reference packing, masked means, clipped ratios, Schulman KL, advantages, and run identity match |
| Online-GRPO schedule/checkpoint binding | **Before: FAIL; after focused fix: PASS** | Before: 16 `CHECKPOINT_GROUP_EXPORT_BINDING` disagreements from empty export lists. After changing only dataset `group_id` filtering: zero v1/v2 disagreements and byte-identical producer/independent schedules |
| Serving/trainer isolation and default legacy pack hashes | **PASS** | No serving import of trainer modules; default no-export pack bytes remain pinned |

Machine-readable results are in:

- [online-evidence-audit-task42-cpu.json](evidence/online-evidence-audit-task42-cpu.json), SHA-256 `7e0db7b93429147c0e8a8f949e5729ff8531cc7098b3a9abb45de8ad4d9e6b6e`;
- [online-evidence-audit-preflight-cpu.json](evidence/online-evidence-audit-preflight-cpu.json), SHA-256 `044dc712cc94b94323c13936cac3c52e2b97a851d8520b019e8af238df19a6ab`;
- [online-evidence-audit-wave64-disagreement.json](evidence/online-evidence-audit-wave64-disagreement.json), the initial comparison against the pre-correction Task #50 summary, SHA-256 `21066988a946d20f66a9bed1645078c762c9248d899819ec79db054af5461287`;
- post-fix [v1](evidence/online-evidence-audit-preflight-fixed-v1-cpu.json) and [v2](evidence/online-evidence-audit-preflight-fixed-v2-cpu.json) reports, SHA-256 `73eab864301c38cd395b931f8de17bf9f0f62f152510b7180a9f62ec1c41462e` and `18fac7f5ca3cc6c02a0e886b3b093565bdb3582179e97d7f8907bee26951a1c3`, both `PASS` with zero checkpoint disagreements.

The corrected summary is [online-grpo-preflight-cpu.json](evidence/online-grpo-preflight-cpu.json), SHA-256 `6760951ed4856dc2c35f1b51390ed2449ff02b894e30d7c2b65a693f0f589698`. It links the original Task #50 identity, pre-fix failure, focused fix commit, and both post-fix reports. The before file remains unchanged: producer byte identity was not silently treated as a semantic checkpoint pass.

## Independence and process boundary

`python/umminfer_online_evidence_audit.py` uses only the Python standard library.
It does **not** import `umminfer_online_rollout`,
`umminfer_behavior_token_alignment`, `umminfer_online_grpo_preflight`, their
scoring/objective implementations, or any ML framework. Canonical JSON, strict
duplicate/non-finite rejection, SHA-256 domains, raw-SSE parsing, CAS assembly,
NFC/source-unit reconstruction, reward/advantage arithmetic, packing, and
objective/checkpoint identities are independently implemented in that file.

The test parent uses the accepted fixture builders only to create inputs. Every
verdict then runs in a new `python3 -I` process with rollout, alignment,
preflight, dataset-scoring, GRPO, Torch, Transformers, NumPy, and MLX imports
blocked. Two independent invocations produce byte-identical reports, including
the failed preflight report. A malformed or re-signed input exits 2 and creates
no report. A well-formed disagreement exits 1 and writes the precise independent
versus accepted fields. No evidence file is edited by the auditor.

## Independent reconstruction coverage

### Plans, requests, attestation, raw transport, and rewards

The audit validates the closed plan schema, signed-int64 seeds, canonical
request bodies, behavior-policy/model/config/sampling hashes, complete schedule
and group order, response attestations, and fresh completed HTTP-200 identities.
It rejects stale, cross-group, duplicate, reordered, missing, mixed
offline/online, cancelled, truncated, non-finite, model-drift, and unattested
records before producing a report.

Raw `.sse` bytes are re-parsed independently. Transport chunks and SSE frames
must cover exact contiguous byte ranges with matching hashes and monotonic
ticks. LF/CRLF blank-line framing, comments/keepalives, multi-line `data:`
fields, one final `[DONE]`, normalized event provenance, terminal legs, visual
slots, output/clean text, and source frame identities are checked. Artifact
base64 chunks are reassembled, binary chunk-digest chains and final SHA-256
values are recomputed, and every referenced raw/media CAS file is verified.

The narrative-routing reward is independently recomputed from expected and
predicted modalities. The auditor additionally binds `ablation.prompt_id` and
`expected_visual_modalities` to the scheduled group, so coherent label
substitution cannot be preserved by replay. Complete GRPO/DPO rows are then
rebuilt in reward/source order, with population-standard-deviation advantages,
zero-variance-to-zero behavior, split/leakage checks, manifest sidecars, and the
complete accepted file set. No arbitrary reward spec is imported.

### Sampled ids, token_export source spans, and masks

For v1/v2 exports the audit independently checks:

- exact source-record/config/export hashes, tokenizer available-file manifests,
  loaded three-file tokenizer digest, model/tokenizer identities, terminal and
  unknown-token exclusions;
- raw SFT-target reconstruction with visual route prompts and explicit
  `source=slot-N` anchors, earlier-image anchor CAS binding, and terminal-leg
  consistency;
- v1 UTF-8/codepoint offsets, v2 NFC units, normalized token bytes, conservative
  original source units, raw fast-tokenizer offsets, source event/frame
  attribution, spans, hash chains, and offline joint-SFT weights;
- sampled ids equal to raw-target ids, binary clean-prose masks, zero masks for
  every route/prompt/anchor token, no supervised visual-slot token, no
  prose/directive straddle, and exact masked clean-byte reconstruction.

The clean-id relation is reported, never repaired. Masked sampled ids may differ
from a clean-context post-hoc encoding when removing a directive changes a
tokenization boundary. The offline positive directive weights remain separate
from online behavior masks.

### Current/behavior/reference packing and objective arithmetic

For prompt ids `P`, sampled response ids `R`, and sampled mask `M`, the audit
reconstructs `P+R`, `-100` prompt/unsupervised labels, and exact response masks.
Current/reference policy identities are independently bound to that packing;
neither is silently replaced by behavior. Their logprobs remain explicitly
synthetic arithmetic fixtures, not model outputs.

For supervised positions it recomputes masked means with `math.fsum`, clamped
log ratios, clipped surrogates, reference-minus-current Schulman KL, member and
group losses, run identity, split steps, and every checkpoint cursor. Positive
clipping, negative-advantage min-selection, zero reference KL, and the `exp(-1)`
KL oracle are covered in both token-export formats and in the no-export boundary
mode.

## Retained real Task #42 result

The retained source and additive five-record JSONL files match after removing
only `token_export`. The two accepted cases independently reproduce every
published hash and all 61 sampled ids:

| Case | Sampled | Supervised | Zero directive | Clean relation |
| --- | ---: | ---: | ---: | --- |
| Multilingual text | 24 | 24 | 0 | `EXACT` |
| Interleaved visual explicit marker | 37 | 25 | 12 | `CONTEXT_BOUNDARY_DIFFERENCE_BYTES_EXACT`; 25 masked sampled ids versus 24 clean post-hoc ids, exact masked bytes |

Truncated (`length`), cancelled, and default-off records are retained as raw
negative evidence and excluded from supervision. All are `not_exposed` with no
behavior block. Terminal ids `[248044, 248046]` are absent from accepted sampled
ids.

The original `SHA256SUMS.source` has 43 lines. Only 20 referenced files are
present in the reduced retained fixture; those 20 verify exactly. The other 23
are listed as `not_retained`, not silently counted as rechecked. This audit does
not claim a new 43/43 verification. Its level is **STRUCTURAL**:
`tokenizer_reencode=NOT_RUN` and new numerical logprob recomputation is
`NOT_RUN`. The earlier strong tokenizer report and retained Task #42 forward
`max_abs_logprob_diff=0.0` remain historical evidence, not new work here.

## Wave 64 evidence correction — no silent identity reuse

The accepted Task #50 summary pins `umminfer_behavior_token_alignment.py` at
35,117 bytes / `854e1214b3235c01cfa6f9584fa5588aac786dd1e82f2711c67b2bae406030a7`.
Wave 64 retains the structural bundle-only mode added during integration, so the
current file is 36,652 bytes /
`5150f00b2335a84860d4e2a51ec98b9fbe041568303c7288cd3c01506035bac9`.
This changes source fingerprints and all run-derived identities even though the
accepted source plan, trajectories, packs, group statistics, and objective values
are unchanged.

| Object | Task #50 summary | Wave 64 independently regenerated producer |
| --- | --- | --- |
| Run id | `43d4bd01c7ed6736313af3aa71aae638bef9964c79e6ff5fbdba245371a84a21` | `b235d83e1941c53e506de351a68430488d9ac26b083b01bf43f0153861876243` |
| `run_identity.json` | `7e9e229308c2b7ee22e467b7c762073d76b4a44291a1fd3abef2ef058eb55277` | `6d5e83933ae5314e9f4d15b1fd35f51b1ee54b25e6bc91ef46e221ea5adb2c01` |
| Producer `schedule.json` | `5673567fc9bd75010e20ef430098964cfef40df8ae15207d0d1a2cf2ebbc6dd1` | `5bee84a4fb301c25e041917ef10df71aea1a3964beabb0c29d8c64c0803a67c4` |
| `objective_fixtures.json` | `337b69484cd9b4473e7255fcb0e69c40516d6482d9b62f6175140e54a2adffe2` | `bc14db422b1fa0965c3d1250b73e87ac5731137f3ca1ca4d0d855330da9369df` |
| Producer `manifest.json` | `7f67d81b0232ac1dbd002d1e7f7800a96d7d04a9342400e2ed2e7a3985d386cd` | `0a8c70856886e77b0afdbec8d779a21a3aa01ea82449dca60d73534a51b47a65` |
| Producer `SHA256SUMS` | `276b21f4523a3ae4a6b83a1a6be1f67bf4b0a8ed892fa6ddcef36f29409a23a4` | `20a38d54e04f76ee5ecb2707a9d6452041c6296c58e293c6fb536bd36fa59e9c` |

The corrected summary preserves the old run id and both fingerprint versions in
`independent_audit`; it does not alias the old and new run ids. The config hash
`47ac0f75ada2ec92da7a322a1453903f6ae32ac5ce4c4b91ec44ae933a96ab91`, plan hash
`e3d1629fa9b334620886c5dbfd6c4c2cb402de6e88ba94974f24419572ba46e5`, schedule
hash `8697e83da9be41b56b5bb5985a627c4441dd53d3c866d492ee97c6c20c408bcc`, and
accepted source file-set hash
`de31842ba422edea7662dc29572a1d081fef39b93d29a5f727585c783a36ae31` are
independently unchanged.

## Failed checkpoint finding

The integrated `_checkpoint` builds `completed_group_ids` and
`remaining_group_ids` from dataset `group_id` values, then filters identity
groups using `group["schedule_group_id"] in ...`. Those namespaces differ. As a
result, at all four cursors both export-hash fields are:

```text
canonical_hash([]) =
4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945
```

This is not a valid binding of completed or remaining group exports. The
independent reconstruction uses dataset `group_id` for those lists. It produces
16 precise disagreements: six completed/remaining export hashes, four derived
checkpoint ids, and six step checkpoint links. The corrected schedule is
`0a612bad60c7e0705dfa6109281fe2c904475ae380300284b6fc96618791431b`; the
corresponding corrected manifest and `SHA256SUMS` are respectively
`6ae1b6441708dbc7a5442cacbfc8c419c8bf0b54a9403548ce5623b462b0f97c` and
`2c469d8fcf90574a18d50af22d4f18a7d2fa48a2a76444c6b997b3e5770ff759`.

Those pre-fix corrected hashes were reported as independent findings, **not**
substituted into the producer bundle. A focused regression was then added to
`tests/online_grpo_preflight_test.py`. Before the fix it failed with remaining
export hash `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
instead of independently expected
`3b897e7cf7ca6cc15ec7b326c561b9e9b691e7516647a17917aa83f4441c030e`.

Commit `ea9155dbd1ebe496dfed4d235f928e828c6d111c` changes only the two filters
to use dataset `group_id`. The regression turns green, and fresh post-fix v1/v2
producer bundles are byte-identical to the independent reconstruction with zero
checkpoint disagreements. Because preflight hashes its own source, the fix also
creates new run identities, intentionally not aliased to either earlier run:

| Result | v1 | v2 |
| --- | --- | --- |
| Post-fix run id | `86a7803e675c9fa813fa1aacf18bf383e788597de61524315c1efd3a534a145c` | `92eed6fcca9b6b52016f7561be6769d05677085630246aae4a6b1f0a0128942e` |
| `run_identity.json` | `52fb50b4d9a56e74d7b3fc42de67753c0a5672525bb9d7947323fbcc2487e7ea` | `e0a59f249cb50fa902f0021ca133848374a57fac9b1024eea9a8798e15d04db6` |
| `schedule.json` | `6f58962a24fcf739fddc180d43775632bba64053d57bb4595b10d15b86cf839e` | `b5c450ad49e9b32f413ce8035c5da1a49a4237a923c40d530bf755a662d790ce` |
| `objective_fixtures.json` | `e7ead49850b82179468b2facac656569c8f6f2a61add7388b5f2e30f2142d2b8` | `1ee372ad37cc9117da224ec5fa99a9e10459c7d964cb62e478b6a40987bb1ec5` |
| `manifest.json` | `bc6f40cdbe3a651235ae245c02bebc3b7e6f08e1301bcc20d7820f9fe4b86510` | `a148c4aafcebf89b38f9a02f4fefe337b224905e408b42c490244b07df52ab44` |
| `SHA256SUMS` | `21b7d79b4c48c08e3d415982c83b4fa8a914f688e7bd4cb1bb33bffd335ea57e` | `a9d2c0c08a28c24aaa42e691574cdf101c0b63f82d764de5fc15c3637fe13f53` |

The before failure report remains sealed separately. Online optimizer execution
was and remains `NOT_RUN`.

## Default legacy pack regression

A default no-export SFT pack built from the retained original Task #42 source,
with no online provenance and no additive token exports, remains pinned at:

| File | SHA-256 |
| --- | --- |
| `sft_train.jsonl` | `95aec6d34f0a27695d15856430299f4b95486ab98cf1e8008eb6e20502b4c3aa` |
| `sft_eval.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `manifest.json` | `4d069bfd013a87640981c887e9b5cec5c25d9acf4776a4c6cac481fefc7c4974` |
| `manifest.json.sha256` | `d69cbc49efe50ef8fb1190ef16f8bc7cedc2cf0a0a0199e122ac88de64ac4f0c` |

## Commands and observed results

Independent fresh-process suite:

```sh
python3 -I tests/online_evidence_audit_test.py
```

Observed: 9 tests, `OK`. It covers v1/v2 multilingual, decomposed-NFC, emoji byte
fallback, spoken spans, visual slots, explicit anchors, zero variance, no-export
boundary, retained real Task #42, accepted-evidence mismatch reporting, and
no-output rejection for stale/cross-group/duplicate/reordered/missing/
NOT_COVERED/cancelled/truncated/non-finite/tokenizer/model-drift/raw-CAS/
re-signed/offline-online-mix inputs.

Related CPU regressions:

```sh
python3 -I tests/online_rollout_test.py
python3 -I tests/behavior_token_alignment_test.py
python3 -I tests/online_grpo_preflight_test.py
python3 -I tests/token_export_test.py
python3 -I tests/sse_replay_test.py
python3 -I tests/sft_dataset_test.py
python3 -I tests/joint_trainer_test.py
python3 -I tests/joint_grpo_live_test.py
python3 -I tests/rl_architecture_test.py
umm-test-local
```

Observed after the focused fix: independent audit 9/9, online rollout 11/11,
online preflight 6/6, behavior-token alignment 9 tests `OK (skipped=1)`, token
export 10 tests `OK (skipped=1)`, raw-SSE replay 9/9, dataset 8/8, joint planner
6/6, joint GRPO 4/4, RL architecture 16/16, joint SFT 6/6, joint DPO 10/10, and
the behavior live-harness contract `PASS`. `umm-test-local` exited 0, including
its C++ unified, realtime, voice, and online-attestation coverage. Both skips are
the optional real-tokenizer cases and remain explicit `NOT_RUN`. The full local
command is CPU-only and does not contact the resident service.

## Task #60 attested-retry correction

The first real Task #49 rollout completed on attempt 2 after two genuine HTTP
503 admissions. The independent auditor originally hard-coded attempt 0 in both
provenance and response-attestation reconstruction, so its fresh Task #60 audit
rejected that valid record with `STALE_OR_CROSS_GROUP_PROVENANCE`
(`independent=2 evidence=0`) before any new GPU pass. The failed R3 bundle is
retained separately; the accepted Task #49 evidence was not rewritten.

Commit `b2f6931d5a7ffe235d7a9b9579f55349cb5cbf60` now derives the attempt from
accepted provenance, requires an integer within the plan's retry bound and a
retry-enabled policy for nonzero attempts, and reconstructs the response
attestation with the identical attempt. A focused v2 fixture proves attempt 2
byte-identical reconstruction and rejects a consistently re-attested,
out-of-policy attempt 5 with no report. The independent suite now passes 10/10;
default attempt-0 v1/v2 fixtures and the sealed pre-fix checkpoint/group-id
records remain unchanged. This correction does not execute an optimizer or make
a model-quality claim.

The next real-model audit also exposed a separate fixture-size assumption: the
exact token-export root contains 29 regular files, including all 18 multi-GiB
weight shards, and `file_identity` used the generic 64 MiB `read_file` cap. R4
therefore rejected the first shard as `FILE_TOO_LARGE`, again before GPU use.
Commit `37b4f465a65d3d1a8369fd70e32820a3ca89ac69` adds bounded 8 MiB streaming
SHA-256 for every declared tokenizer/model file while leaving the generic cap on
small JSON/SSE evidence intact. A focused test covers a shard larger than 64 MiB
and the independent suite passes 11/11. No weight or tokenizer file is excluded
from the available-files manifest.

A further real-Qwen schema check showed that `added_tokens_decoder` entries can
omit an inner `id` and carry the terminal id only as their JSON member key.
R5 rejected this valid declaration as `TERMINAL_TOKEN_IDENTITY`, before GPU use.
Commit `1894a66c7513b69d751d74d4f049808354d7153d` mirrors the accepted
validator's `value.get("id", key)` fallback. A real-shaped v2 fixture now binds
`[248044, 248046]` correctly, and the independent suite passes 12/12. EOS remains
excluded from sampled supervision; this is schema compatibility, not an EOS
validation bypass.

## Explicit NOT_RUN / NOT_COVERED boundaries

- New GPU, inference, serving, optimizer, model-quality, and production-training
  outcomes: **NOT_RUN**.
- Current/reference policy forwards and new numerical behavior-logprob
  recomputation: **NOT_RUN**. The objective values are deterministic synthetic
  fixtures only.
- Real attested online generation: **NOT_RUN**. Deterministic fake-endpoint
  bundles and retained Task #42 captures are not new online generation.
- Real local Qwen tokenizer re-encoding and a new strong tokenizer gate:
  **NOT_RUN**. Structural masks/spans are not fresh tokenizer proof.
- Original Task #42 files absent from the reduced fixture: **NOT_RETAINED** and
  not independently rechecked, never marked as 43/43 PASS.
- Real optimizer/tensor/RNG checkpoint state: **NOT_RUN**. Preflight checkpoints
  are identity descriptors only; their former group-export defect is retained as
  before evidence and the two-line identity fix has only CPU contract coverage.
