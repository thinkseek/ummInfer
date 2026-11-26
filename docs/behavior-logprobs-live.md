# Native Qwen3.8 behavior-token/log-probability evidence gate

Date: 2026-11-26. Accepted implementation/fix commits:
`a2c7df50f3d203ecfee0b757d5d43e2b18f04d6a` and
`e1360aad91add49ebf72e1f628223779ac545fd0`. The final isolated run used the
second commit exactly. This is a serving/evidence-transport and numerical
recomputation result, not an optimizer, model-quality, or generalized
performance claim.

## Producer contract

`POST /v1/chat/completions` accepts `"behavior":{"logprobs":true}` only with
`stream=true`. Default requests and explicit false/null options keep the old
stage parameters and public wire shape and emit no behavior member or
behavior-evidence response header. The opted-in response is terminal-buffered
so `X-UmmInfer-Behavior-Evidence: exposed|not_exposed` is selected only after a
complete terminal record is known. Required online rollout request headers are
echoed on every response, including validation failures. Non-opted-in streaming
continues through the historical chunked provider.

A covered last SSE data frame has exactly `format`, `tokenizer_sha256`,
`token_ids`, `logprobs`, and `mask`, followed by `[DONE]`. Scores are full
log-softmax over the exact FP32 native LM-head row used for that token's
selection, before temperature scaling or top-p filtering. Greedy decode retains
device argmax; stochastic decode consumes the existing single RNG draw. EOS and
lookahead tokens are absent. No text is re-encoded to manufacture ids.

Narrative route tags/prompt bodies/anchor suffixes are actual sampled tokens but
have `mask=0` only when retained token bytes reconstruct the exact parser-clean
prose. Invisible tokens are also zero. Generated media bytes and
server-synthesized fallback prompts/anchors are absent from the sampled
sequence. A prose/directive-straddling token, malformed evidence, unavailable
logits, missing tokenizer identity, batching, unsupported audio/video topology,
truncation, cancellation, or empty output fails closed as `not_exposed` with no
behavior block.

The tokenizer digest uses the domain-separated
`umminfer-qwen38-loaded-tokenizer-v1` manifest over exact byte length and
SHA-256 for `merges.txt`, `tokenizer_config.json`, and `vocab.json`, hashed from
the same buffers parsed by the tokenizer. It is intentionally distinct from the
`token_export` available-files digest.

## Exact provenance

| Object | SHA-256 |
| --- | --- |
| Source commit | `e1360aad91add49ebf72e1f628223779ac545fd0` |
| Gateway binary | `344da9bb9775c4dfe1af1db727baf641e1d1bf0301789014631d04ed964936a3` |
| Native Qwen stage | `5284cf5f7f405c33873d227a13f2c12240d84e61fb7740bd0ed92160c46cf1ea` |
| Fresh native replay binary | `1f582fe82d3f638d23a600caedb14816c151e8993b8181ed6c5b8a1718014cde` |
| Model `config.json` | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| Canonical 18-shard model manifest | `e38893095032f8fb1115c76a2e264acb8a13c5233ee69ad3fd73d46c27386613` |
| Exact loaded three-file tokenizer manifest | `1dd1199e35f6cfb9f547458e02c9fb3a522e9c346ed22ba3d461aeed14310016` |

The model declares `Qwen3_5ForConditionalGeneration` / `qwen3_5`; its upstream
revision is `UNKNOWN`. All 18 shard hashes, 11 configuration/tokenizer file
hashes, and 10 source-file hashes are preserved in
[the machine-readable evidence record](evidence/behavior-logprobs-live-sm90.json).

The accepted run used an ephemeral loopback endpoint, serial
`umm-native-stage-qwen38 --gpu 7 --batch-size 1`, and two CPU-only contract
fixtures. The visual output PNG is a contract fixture, not generated-model or
perceptual-quality evidence. Physical GPU 7 was the only CUDA device used;
GPUs 0/1 and the resident GPUs/service were not used or restarted.

## Commands

Builds used Zig ReleaseFast and CMake Release with CUDA 12.4 / SM90 in isolated
source and build directories. The live command, with paths intentionally
expressed as placeholders, was:

```sh
UMMINFER_LIVE_OPT_IN=1 python3 -I tests/behavior_logprobs_live.py \
  --model qwen3-native --model-dir <model-root> \
  --server-binary <isolated-gateway> \
  --stage-binary <isolated-qwen-stage> \
  --replay-binary <isolated-native-replay> \
  --source-commit e1360aad91add49ebf72e1f628223779ac545fd0 \
  --out-dir <new-empty-evidence-dir> --gpu 7 --timeout 600
```

The harness starts only its gateway, serial Qwen stage, and two CPU fixtures.
It stops those recorded PIDs before starting each fresh native replay process,
so two 51 GiB model instances never coexist on GPU 7. It requires exact sampled
ids, masks, tokenizer identity, public clean-target equality, and finite scores
within absolute tolerance `1e-6`; the measured maximum difference was `0.0` for
all 61 exposed tokens.

## Captured streams and fresh forward recomputation

| Case | Coverage / terminal | Raw SSE bytes / SHA-256 | Tokens (masked) | Fresh-process result |
| --- | --- | --- | --- | --- |
| Multilingual stochastic text (`temperature=0.7`, `top_p=0.9`, `seed=42`) | `exposed`, completed | 7,805 / `0301732d82bbb319a634c929ac99e0e87994af216d80e1e49f96eb3c36272a58` | 24 (24) | Exact ids/masks; max abs score diff `0.0` |
| Greedy interleaved image narrative (`seed=-42`) | `exposed`, completed | 239,496 / `6d90da4e04695ec1c64a0be619dac73d127f7fd9524ddcf41fecfca3eb210b18` | 37 (25) | Exact ids/masks; 12 directive tokens excluded; max abs diff `0.0` |
| One-token truncation | `not_exposed`, completed with `finish_reason=length` | 781 / `df62e656957a1fe07ebd9779601d3e68d3b65335264736554af4fce07e13116f` | null | No behavior block |
| Real in-flight `/abort` | `not_exposed`, cancelled | 5,288 / `305f0768baf01d5124a530e086dc37546843645ea337836be21fb0b114aeda7a` | null | No partial evidence |
| Default-off request | `not_exposed`, completed | 6,995 / `e013163468e3e11aee5acfa92775ae740d568dae69587e2dfa1c15133f82bcd3` | null | No behavior member/header-only opt-in leakage; clean text matched the opted-in text case |

The text evidence canonical hash is
`2ad5e0325359e2759e434310d6461b410107522baf27803252aaa832c30f7d49`; the visual
evidence canonical hash is
`8f7c20fe706978a52fd5c9a1f866574fdcd90d995d6ead67916b6538e4369c07`.
Fresh prompt SHA-256 values are
`754352b1f16d93efdfa7bf84cb63dc1db18a8907f3c3d1f1283f95e461333353` (text) and
`b1eb1ae7f8029913112c7e9dc0bd5052a19cb1ee07a40c14174112253722436a` (visual).
Fresh raw/masked text hashes and every token id/log-probability/mask are bound in
the final evidence directory and summarized in the machine-readable record.

## Raw-SSE replay and independent checksums

A fresh `python3 -I python/umminfer_sse_replay.py` process produced byte-identical
canonical JSONL: 59,351 bytes, SHA-256
`21be90a9641d68962c45ffc60e95689f83bdee8752c88ab335097748a618f2ce`.
The on-disk audit verified all five replay refs, one output media ref, and six
CAS files. Split reads using a `1/2/3/5/8/13` byte pattern preserved raw bytes,
frame offsets/kinds/hashes, normalized events, artifacts, and terminal state:
1,464 text chunks, 44,908 visual chunks, 149 truncation chunks, 994 cancellation
chunks, and 1,313 default chunks.

Independent `sha256sum -c SHA256SUMS` verification passed all 43 final evidence
files. The preserved final `summary.json` SHA-256 is
`23fb40f8bcd659a19386d4451fd7c832ae00db3dbc69947a86a3d0f6457c2ecc`; the
43-line `SHA256SUMS` file SHA-256 is
`1e54fc884687b67c6f7b19b66c75a9ee8624fb58852f21253adda6e42a217539`.

## Failed attempts retained separately

Earlier attempt directories and their summaries/logs remain outside the tracked
tree. They are not presented as accepted evidence. Attempts 1–7 exposed a
harness bug that eagerly consumed the response body in an HTTP 200 assertion;
attempt 8 used a nonexistent validator method; attempt 9 reused a helper with a
hard-coded three-record expectation; attempt 10 passed a `Path` instead of raw
bytes to the framing helper. Each defect was fixed narrowly, with deterministic
fake-endpoint capture and split-framing regressions. No sampled-id,
log-probability, mask, independent-forward, or cleanup assertion was weakened.
Two isolated CUDA build errors (a missing namespace qualifier and an overly
broad prompt link) were also corrected before the accepted run.

## Cleanup and retained limits

The gateway, Qwen stage, both fixtures, and fresh replay processes exited. The
ephemeral endpoint refused connections after cleanup and physical GPU 7 returned
to `3 MiB`. No resident process was signaled. The full
`umm-test-local test`, native/stage CTest, rollout/replay/token-export Python
suites, and `tests/behavior_logprobs_live_test.py` are the local regression
matrix; one pre-existing timing-sensitive nucleus check initially measured
2.063/2.067 ms and passed when rerun without concurrent test load.

This gate does not run an online optimizer, evaluate model quality, claim an
improvement, or make batched/audio/video behavior evidence available. Those
remain `NOT_RUN`/`NOT_COVERED` as applicable.
