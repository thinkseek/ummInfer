# Real isolated online-rollout generation gate

Date: 2026-11-26. Accepted run: attempt 6, source commit
`66ebb54cf920a7482d1b4c62bd610fdf2fcde40f`. The gateway and Qwen stage were
built from `713d58ab401fbdb13ac4624ee2c8b8b620e43729`; the only later C++ change
before this run was the independent replay CLI's auto-narrative option. This is a
real native online-generation and evidence-transport result, not an online
optimizer, model-quality, generalized-improvement, or generated-image-quality
claim.

## Frozen plan and native identity

The frozen plan had two groups of three unique signed-int64 seeds, shared
`max_tokens=96`, `temperature=0.0`, `top_p=1.0`, and
`behavior.logprobs=true`:

| Group | Seeds | Required response |
| --- | --- | --- |
| ASR-free multilingual text | `101, 102, 103` | Clean text, no visual slot/artifact |
| Interleaved image narrative | `-201, -202, -203` | One supported image slot/artifact and aligned directive masks |

The shared director was `visual_mode=auto`, `max_slots=1`,
`allow_text_only=true`. The image was a **CPU contract fixture**, not a generated
model artifact. Transport allowed at most five fresh attempts per rollout, with
retry only for HTTP 503. The complete canonical plan, request bodies, group ids,
and response attestations are in the evidence bundle.

The real Qwen stage wrote `umminfer-native-stage-identity-v2` only after loading.
It independently hashed `config.json`, all 18 available safetensors shards, and
its executable, and recorded pid/start-time and the exact serial stage config.
The gateway independently rehashed the same files and required the attached live
VLM's `stage_ready` digest to equal the pinned document digest before consuming
an attempt. No client-supplied loaded manifest was accepted as proof.

| Object | SHA-256 |
| --- | --- |
| Gateway binary | `cbf8b7b01a4bd06511f1a08e3d39c869e05d74de8e30dac93d083895bbda5657` |
| Native Qwen stage | `857c5b2fa1bc7317007fcb7d55bc18aaa04fc026b556f518d66ed1c4dc41e5f1` |
| Independent auto-narrative replay binary | `ae1aa8e722a2c9aa2c48ca91b5fabf46a9f2a73b6b6f5fa9f91a4a26d5b40bca` |
| Model `config.json` | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| Canonical 18-shard weight manifest | `e38893095032f8fb1115c76a2e264acb8a13c5233ee69ad3fd73d46c27386613` |
| Exact loaded three-file tokenizer manifest | `1dd1199e35f6cfb9f547458e02c9fb3a522e9c346ed22ba3d461aeed14310016` |
| Semantic plan | `b461af977ff04ed7e156d1fde5b94233dfd990479095c96fdd8442fd7282b8da` |
| Schedule | `45188f2d4970d9f0f219e83b982b84a2d3a2ab947fef08dd697f14c8e1c4ce46` |
| Behavior policy | `1f45a7ec17a880338c3713264b5ea9d3b83966b0f45d575a5e6280022b49b513` |
| Sampling | `e8e4f0882cbc49548572f140e71f02b6bb73cd872dca01209c6b50a89b7fd7af` |

The model declares `Qwen3_5ForConditionalGeneration` / `qwen3_5`; its upstream
revision is `UNKNOWN`. Every individual weight, configuration, tokenizer, source,
and binary hash is preserved in
[the machine-readable evidence record](evidence/online-rollout-live-sm90.json).

## Actual client, fresh attempts, and accepted groups

The accepted generation used the actual trainer-side client:

```sh
python3 -I python/umminfer_online_rollout.py run \
  --plan <frozen-plan> --endpoint <isolated-loopback> --out <new-accepted-bundle>
```

A real long-running request held the single admission slot. The scheduled first
rollout therefore received two fresh, attested HTTP 503 responses. Cancelling
only that task-owned blocker (`http-1`) allowed attempt 2 to succeed. All eight
attempt records were preserved in order: `503, 503, 200, 200, 200, 200, 200,
200`. The other five accepted rollouts succeeded on attempt 0. No failed or
cancelled response entered a group.

| Group / seed | Public request | Attempt | Tokens (masked) | Raw SSE SHA-256 |
| --- | --- | ---: | ---: | --- |
| Text / `101` | `http-4` | 2 | 24 (24) | `b2a6b7055b4da9f5c8c58f17bef9aa63e8ec1bc18c5c00ae17776242be844a3d` |
| Text / `102` | `http-5` | 0 | 24 (24) | `af9c126bc37099832b65813f2b3080f8fe13c9583450a6f2218c0f1156bcc23a` |
| Text / `103` | `http-6` | 0 | 24 (24) | `3d4953f20065fafe90660ac69811de0f8e0cebb81fad7cab4bdb9723eceeb8bf` |
| Image / `-201` | `http-7` | 0 | 40 (28) | `0137f09de6f7a9589844d0c0508384a619e55c75a3d8ee6a880e809f2334abcd` |
| Image / `-202` | `http-8` | 0 | 40 (28) | `881e332145ae61ea0738781d1a4c0cfad143a6af3c4dce79936efb45bd110cb1` |
| Image / `-203` | `http-9` | 0 | 40 (28) | `c955c7638557c3619833a45b1726c004346115dcdb29479e752ab8d1103814d5` |

The text behavior evidence canonical SHA-256 is
`633624dfc73c840ef3c81b40e3e1c9d2ae5f5a105ff2824b816fcc3b87f152e4`;
the image evidence canonical SHA-256 is
`f4b4f8204d4381b87cecb2354e48c6959391bd145532d4a8e05adb0c6bad5fe2`.
Greedy sampling intentionally produces the same evidence within each group, while
request/session/rollout identities remain unique. Rewards were computed only
from the exported trajectories. The six-record trajectory is 127,955 bytes with
SHA-256 `d7b114285d54271c24da39180584c59d19476322f0142cc10be2f801ef9b05ad`;
the GRPO pack manifest SHA-256 is
`501e3a2f8386bacdd4e97852e5ad04de876030995ebf7b4ef8d35da553aaf6e8`.

## Independent native numerical gate and replay

After the serving processes exited, six fresh native processes independently
recomputed the exact auto-narrative prompt prefixes and all 192 sampled tokens:

- every sampled token id matched;
- every clean-target mask matched (156 supervised tokens total);
- every public clean-text target matched the retained sampled bytes;
- the maximum absolute full-logprob difference was `0.0` for every token, under
  the required absolute tolerance `1e-6`.

The image group excluded 12 directive tokens per response. No offline source-span
validator was used as a substitute for this native forward gate.

Fresh-process `python3 -I python/umminfer_online_rollout.py replay` reproduced
all 17 accepted files byte-for-byte, including trajectories, pack files, raw-SSE
CAS, and media CAS. The separate independent verifier then rehashed the model
config and all 18 weights, repeated the accepted-record and negative-case checks,
and produced another byte-identical replay. Final `sha256sum -c SHA256SUMS`
verified all 108 evidence files.

| Verification object | SHA-256 |
| --- | --- |
| Live `summary.json` | `5298574cf84c707815d6d73d6caf9fc32842502297d954f0caee5a46a4ed903f` |
| `independent-verification.json` | `72975c89b8bef595a751636308d9d7e33cd72fa4c44bcff66d444b921ad7b599` |
| Final 108-line `SHA256SUMS` | `0b340a818c15e6699a2bd22858991aed9f0890559d2e9cf4f7f1c76a2aed87f1` |

## Negative coverage and cleanup

| Case | HTTP / terminal | Coverage and group eligibility |
| --- | --- | --- |
| Default protocol, behavior option absent | 200 / completed | No behavior header/block; output matched accepted text; not accepted |
| One-token truncation | 200 / `finish_reason=length` | `not_exposed`, no behavior block, not accepted |
| Real in-flight `/abort` (`http-12`) | 200 / cancelled | `not_exposed`, no partial evidence, not accepted |
| Unsupported video topology | 400 / failed | `not_exposed`, no behavior block, not accepted |

The retry blocker also preserved a cancelled raw-SSE record with SHA-256
`c85709bbd795f8ee385f7f66fc1ca42373a79430d7b58fe42842edc5b23b87fb`.
`/health` reported `in_flight=0` before shutdown. The gateway, native Qwen stage,
both CPU fixtures, and all six fresh forward processes exited. Physical GPU 7
returned to `3 MiB`; no other GPU or resident service was contacted or restarted.

## Earlier failed attempts retained

Attempts 1–5 and their logs/documents remain separate from accepted evidence:

1. The launch recorder used a log handle instead of its path; fixed before model
   generation.
2. The harness omitted the required `legacy` readiness fixture; stopped before
   scheduled generation after `/ready` remained 503.
3. The server emitted a valid exposed stream, but the client incorrectly rejected
   the optional absent inner `request_id` on ordinary unified events. Validation
   now checks it when present while always requiring outer id/session identity.
4. Raw-capture preconsumption could not prove early blocker overlap; stopped
   before scheduled generation and replaced with a real `/health` admission gate.
5. The image-only director could contaminate the text group; stopped before
   scheduled generation and replaced with matching auto-director generation and
   independent replay.

No acceptance assertion was weakened to turn those failures into success.

## Local regression matrix and retained limits

`umm-test-local` passed in Debug and ReleaseFast. The final matrix also passed:
13 online-rollout tests, 16 RL architecture tests, the behavior harness test, 9
raw-SSE replay tests, 9 native stage CTests, 3 native root CTests, 8 SFT dataset
tests, 6 Stage C pipeline tests, 6 joint-trainer tests, 4 frozen offline GRPO
preflight tests, 38 controlled-measurement tests, and the 70-case multimodal
replay. The optional real-tokenizer fixture remained `NOT_RUN`. The separate
stability suite was environment-blocked by missing pinned `jieba 0.42.1` and was
not counted as passed. One timing-sensitive nucleus benchmark initially measured
`5.500 ms/token` under concurrent load, then passed unchanged on quiet reruns;
the full native root suite passed 3/3.

This gate does not run an online optimizer, evaluate model quality, or claim
improvement. Audio/video behavior evidence and batched behavior evidence remain
uncovered. The accepted pack remains input to a future separately gated trainer;
it is not itself a training result.
