# Sampled behavior ↔ token_export CPU alignment gate

Date: 2026-11-26. Implementation/tests commit:
`c86bb47d3125f75e7a00652021c4deda8e07f4cc`.
This is an independent CPU-only validator over retained Task #42 captures and
new deterministic fixtures. It does not run a server, model, GPU, optimizer, or
fresh numerical forward pass. New outcomes beyond the retained captures remain
`NOT_RUN`.

## Contract and evidence flow

`python/umminfer_behavior_token_alignment.py` reads a pinned
`umminfer-behavior-token-alignment-fixture-v1` bundle. It independently combines
the existing raw-SSE and token-export primitives without calling the online
rollout's whole-record acceptance routine or changing that routine.

Before any optional tokenizer import or report output, it requires:

- canonical strict JSON/JSONL with no duplicate members or non-finite numbers;
- exact source and aligned trajectory hashes, unique ordered request/session
  identities, and equality of every aligned source record after removing only
  its additive `token_export`;
- exact request JSON, response attestation, policy/model/config/sampling hashes,
  raw-SSE content length or chunked transport, outer request/model ids, and
  non-conflicting inner session/request ids;
- raw `.sse` bytes, chunk/frame offsets, source-event hashes/ticks, one final
  terminal, final `[DONE]`, media CAS bytes and SHA-256 values;
- exactly one final `umminfer-behavior-logprobs-v1` block for accepted cases,
  complete non-empty equal-length ids/logprobs/masks, finite non-positive scores,
  binary masks, and exact raw/summary/native-replay evidence equality;
- completed `finish_reason=stop` evidence only; truncation, cancellation,
  default-off responses, and every `not_exposed` response stay out of
  supervision;
- token_export v1/v2 source-record, policy, tokenizer identity, complete
  available-file hashes, normalized/original UTF-8 units, source events/frames,
  spans, raw offsets, and existing joint-SFT mask integrity;
- the domain-separated `umminfer-qwen38-loaded-tokenizer-v1` digest recomputed
  from the exact loaded `merges.txt`, `tokenizer_config.json`, and `vocab.json`
  bytes, plus generation/tokenizer/added-token EOS declarations;
- no terminal/EOS or unknown token in sampled supervision, exact sampled ids
  versus post-hoc re-encoded **raw SFT-target** ids, and clean-prose bytes after
  applying the sampled mask;
- `mask=0` for every visual route tag, prompt body, and explicit
  `source=slot-N` anchor token. Prose/directive-straddling tokens and supervised
  tokens mapped to a visual-slot event fail closed. The offline `joint_full`
  SFT weights remain a separate, already validated mask; their positive
  directive weights are not relabeled as online behavior masks.

Only after those checks does `--reencode` load the local fast tokenizer and
independently re-encode both token_export streams. Fresh raw replay then
rebuilds source and aligned JSONL plus CAS in a temporary directory and requires
byte identity. `--report` refuses an existing path and writes nothing on any
failure.

## Retained real Task #42 result

The read-only fixtures are under
`tests/fixtures/behavior_token_alignment/task42/`. The original source JSONL is
kept separately from the additively token-exported JSONL. The real strong gate
used the exact six-file local tokenizer directory with Transformers 5.6.0 and
tokenizers 0.22.2, with no model weights loaded by the validator.

| Case | Sampled ids = raw-target re-encoding | Mask | Clean-prose re-encoding relation | Result |
| --- | --- | --- | --- | --- |
| Multilingual text | 24/24 exact | 24 supervised | 24 masked sampled ids = 24 clean post-hoc ids | `PASS` |
| Interleaved visual | 37/37 exact | 25 supervised; 12 directive tokens zero | **Different:** 25 masked sampled tokens versus 24 clean post-hoc ids at the removed-directive context boundary; masked bytes are exact | `PASS` |
| Truncated | not applicable | no evidence | `length`, `not_exposed` | excluded as required |
| Cancelled | not applicable | no partial evidence | cancelled terminal | excluded as required |
| Default off | not applicable | no evidence | completed, `not_exposed` | excluded as required |

The visual clean-id difference is not repaired by substitution. Sampled
behavior remains bound to the 37-id raw target; the separate 24-id clean stream
is never presented as the sampled sequence. Masked sampled token bytes, source
spans, visual-slot boundaries, and clean NFC prose all match exactly.

The report records `tokenizer_reencode=PASS` and
`raw_sse_fresh_replay=PASS_BYTE_IDENTICAL`. Numerical logprob recomputation is
`NOT_RUN` in this task. The retained Task #42 fresh-forward record still reports
`max_abs_logprob_diff=0.0`; this gate binds that record but does not claim a new
forward result.

### Exact hashes

| Object | SHA-256 |
| --- | --- |
| Original five-record trajectory JSONL | `21be90a9641d68962c45ffc60e95689f83bdee8752c88ab335097748a618f2ce` |
| Additively token-exported JSONL | `5d8a36a82c9dcacaad5eb7263fbd1b24994b3d46effc2f5d0cad70d97869e26d` |
| Fixture `alignment.json` | `94292293fe4e03e5dd188019c92ab0c1d8130408dd7af3614ecc417f78b1f945` |
| Original Task #42 `summary.json` | `23fb40f8bcd659a19386d4451fd7c832ae00db3dbc69947a86a3d0f6457c2ecc` |
| Original 43-line `SHA256SUMS` | `1e54fc884687b67c6f7b19b66c75a9ee8624fb58852f21253adda6e42a217539` |
| Text raw SSE | `0301732d82bbb319a634c929ac99e0e87994af216d80e1e49f96eb3c36272a58` |
| Visual raw SSE | `6d90da4e04695ec1c64a0be619dac73d127f7fd9524ddcf41fecfca3eb210b18` |
| Text behavior evidence | `2ad5e0325359e2759e434310d6461b410107522baf27803252aaa832c30f7d49` |
| Visual behavior evidence | `8f7c20fe706978a52fd5c9a1f866574fdcd90d995d6ead67916b6538e4369c07` |
| Text source record | `50fceb2e515a59316bee4276085faf1ebb29ccfb21f790da0c660394c4c8e351` |
| Visual source record | `ff5390f918dee0909c1a2953803e5dd69665ff55279796127821423fb35cf2a9` |
| Text token_export | `89f86522d0199d1c85a7e206aecc28e78569cb2aaaab91392dae293aae7ec1d3` |
| Visual token_export | `174b7363735533ccfb2f0838e6d937fe9f09e7b56bb3d46e716f15dbb3d40973` |
| Loaded three-file tokenizer manifest | `1dd1199e35f6cfb9f547458e02c9fb3a522e9c346ed22ba3d461aeed14310016` |
| Six-file token_export available-files manifest | `c23e55ebc38d9d210299a84a602ffd2edfc8ee646e39cbd103677e89e33ee60b` |
| Strong fresh-process report | `428cb273647e2da837b86407d887708cb3fed430262c5aa4319345720bdeca60` |
| Fresh-process tamper report | `835db88bee381d8056c6a5081353072b21ef251a1d27f6d5e100bc166bc17313` |

The six exact tokenizer files are:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 4,312 | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| `generation_config.json` | 202 | `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e` |
| `merges.txt` | 3,353,259 | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |
| `tokenizer.json` | 12,809,320 | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| `tokenizer_config.json` | 9,984 | `207c03c41a06a9c3b0cc527d8a56115159c1eac6b59a4ffae85b646e773d074e` |
| `vocab.json` | 6,722,759 | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |

Terminal ids `[248044, 248046]` are absent from all 61 sampled ids.

## Commands and observed results

Default deterministic v1/v2, real-fixture structural regression, and
fresh-process CLI/no-output checks:

```sh
python3 -I tests/behavior_token_alignment_test.py
```

Result: 9 tests, `OK`, one explicit real-strong skip when no tokenizer directory
is supplied. Valid v1 covers an explicit visual anchor; valid v2 covers
multilingual NFC composition, combining marks, emoji byte fallback, spoken
spans, and visual slots. Negative fixtures cover stale, truncated, cancelled,
reordered, duplicate, cross-record, tokenizer-mismatched, target-mismatched,
non-finite, fabricated-value, EOS, mask/source-span, raw-SSE, and CAS failures.
Every invalid CLI case exits 2 before an optional ML import or report output.

The reproducible real strong and tamper gate is:

```sh
python3 -I tests/behavior_token_alignment_test.py \
  --real-tokenizer-dir <exact-six-file-tokenizer-directory>
```

Observed result: 9 tests, `OK`, in 22.044 s. It compares a fresh strong report
byte-for-byte with `docs/evidence/behavior-token-alignment-real-cpu.json`, then
runs raw-SSE, PNG CAS, token_export, summary-logprob, and tokenizer-file tamper
cases. Every tamper case exits 2 with no report. The machine-readable tamper
result is `docs/evidence/behavior-token-alignment-tamper-cpu.json`.

The standalone fresh-process CLI contract is:

```sh
python3 -I python/umminfer_behavior_token_alignment.py \
  tests/fixtures/behavior_token_alignment/task42 \
  --tokenizer-dir <exact-six-file-tokenizer-directory> \
  --reencode --report <new-report.json>
```

Related regressions all passed:

```sh
python3 -I tests/token_export_test.py
python3 -I tests/online_rollout_test.py
python3 -I tests/sse_replay_test.py
python3 -I tests/behavior_logprobs_live_test.py
python3 -I tests/rl_architecture_test.py
umm-test-local test
```

Observed: token-export 10 tests (`OK`, one real-tokenizer skip), online-rollout
11, SSE replay 9, RL architecture 16, behavior harness `PASS`, alignment 9, and
the full local C++/voice/realtime/unified test command exit 0. No GPU or
resident service was used.

## Requirements before an online optimizer may consume this evidence

This CPU gate is a data-alignment boundary, not an optimizer result. A future
online optimizer must still:

1. independently verify the actual behavior model, policy revision, weights,
   config, sampling, request/seed/group membership, and response attestation;
2. retain exact raw SSE and CAS evidence, reject cancellation/truncation/stale,
   duplicate, reordered, cross-record, or incomplete groups, and prove
   fresh-process replay;
3. bind the same loaded tokenizer files and terminal ids, preserve actual
   sampled ids separately from every post-hoc encoding, and never substitute
   clean-context re-encoded ids for sampled behavior;
4. revalidate clean/source spans, NFC/original units, directive/prompt/anchor
   zero masks, visual-slot provenance, EOS exclusion, and complete finite
   logprobs before ML import or optimizer output;
5. independently establish numerical behavior-logprob correctness against the
   attested behavior model. Hash-bound retained scores are not a new numerical
   proof; and
6. run a separately accepted online optimizer and model-quality/generalization
   evaluation. Those outcomes, new inference, new numerical forwards, serving,
   and GPU execution remain `NOT_RUN` here.
