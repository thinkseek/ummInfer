# Multimodal benchmark system

This document defines the real-data benchmark for the unified engine, the
metrics, the failure classes, and the stability gate. The committed set is
versioned in `benchmarks/multimodal_manifest.json`. The runner is
`tests/multimodal_benchmark.py`; the downloader is
`benchmarks/multimodal_data.py`.

The branch that added this system did not operate SM90. A separate agent
deploys the long-running service. The main controller runs this harness against
that service and reports the evidence. Mock servers exist only in
`tests/multimodal_benchmark_test.py` and are labelled `contract-test`.

## 1. Data

| Task | Dataset | Cases | License | Public source | Pin |
| --- | --- | ---: | --- | --- | --- |
| ASR WER (English) | LibriSpeech test-clean | 16 | CC BY 4.0 | `openslr.trmal.net/resources/12/test-clean.tar.gz` | archive MD5 matches openslr `md5sum.txt`; SHA-256 `39fde525…e6c23`; per-case WAV SHA-256 |
| ASR CER (Mandarin) | FLEURS `zh_cn` test split (config `cmn_hans_cn`) | 16 | CC BY 4.0, declared by the dataset card | `huggingface.co/datasets/google/fleurs` at revision `70bb2e84…94dd` | archive SHA-256 `09d19ad1…fc166` (it equals the Git LFS oid); per-member and per-converted-WAV SHA-256 |
| GSM8K exact | GSM8K test split | 16 | MIT | `raw.githubusercontent.com/openai/grade-school-math/.../test.jsonl` | file SHA-256 `3730d312…39d14`; per-row SHA-256 |
| Image VQA | VQA v2.0 val | 16 | CC BY 4.0 (questions and annotations; the JSON declares it) | `visualqa.org`, `s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/`, COCO val2014 images | question/annotation zip SHA-256; per-image SHA-256 |
| Joint image + real speech | SpokenCOCO via `mteb/SpeechCoco` validation | 6 | CC BY-SA 4.0 per upstream card; COCO images keep Flickr terms | `huggingface.co/datasets/mteb/SpeechCoco` at revision `ba337542…1576` | revision pin plus per-audio and per-image SHA-256 |

Total: 70 cases. Every case has a stable id, task, modalities, input
reference, reference answer, metric, and tags.

Selection rules are recorded in the manifest and are deterministic:

- LibriSpeech: one utterance per chapter, chapters sorted lexicographically,
  first 16 chapters, first utterance per chapter.
- FLEURS zh_cn: `test.tsv` rows in file order, the first 16 rows whose
  `client_id` has not appeared earlier, so each case is a different speaker.
- GSM8K: first 16 lines of the pinned test file.
- VQAv2: validation questions in file order, first 16 with a unique image id.
- SpokenCOCO: validation rows 0..5 at the pinned revision.

The joint set is real. Each row pairs a COCO image with a human recording of a
caption of that same image, and carries the caption text. No audio is
synthesized, and no image and audio are paired at random. The harness does not
download the LibriSpeech corpus from a mirror that lacks the official MD5.

### Coverage matrix

| Capability | Cases | Metric | Verifiable |
| --- | ---: | --- | --- |
| English ASR (`asr_en`) | 16 | corpus WER and CER | yes |
| Mandarin ASR (`asr_zh`) | 16 | corpus CER (and jieba WER) | yes; measured on SM90 in [multimodal-benchmark-results.md](multimodal-benchmark-results.md#173-fix-and-re-test) (CER 0.0916, WER 0.0503) |
| GSM8K numeric answer | 16 | exact integer after `####` | yes |
| Image VQA | 16 | official VQA accuracy | yes |
| Joint image + real speech | 6 | ASR WER plus caption-keyword hit | partial; answer goes to human review |

### Known gaps

- **Mandarin ASR:** covered by 16 FLEURS `zh_cn` cases (CC BY 4.0). The
  archive is 525,346,466 bytes, under the download byte cap, so the previous
  objection (the datasets-server parquet scan limit) does not apply: the
  downloader fetches the audio archive directly and pins it by SHA-256.
  AISHELL-1 and THCHS-30 are Apache-2.0 but 15 GB and 6.4 GB; both exceed the
  byte cap, so neither is used. The 16 Mandarin cases have since been scored
  across three identical 70-case SM90 rounds (corpus CER 0.0916, jieba WER
  0.0503; see [multimodal-benchmark-results.md](multimodal-benchmark-results.md#174-mandarin-wer)).
- **Joint answer quality:** the joint answer is open-ended. The harness scores
  two objective sub-metrics and saves the full answer with
  `human_review_required=true`. It does not invent a quality score.
- **VQA text normalization:** the harness uses lowercase, punctuation removal,
  article removal, and number words 0..10. The official toolkit has more rules.
  This difference is declared and applies equally to every run.
- **Token rate:** `tokens_per_s` uses `usage.completion_tokens` when the public
  response carries it, otherwise a supplied tokenizer, otherwise a
  whitespace-word estimate. `tokens_source` labels the basis; an estimate is
  never reported as an exact token count.
- **COCO image terms:** VQA and SpokenCOCO images come from Flickr. The image
  license is not uniform. The manifest records the dataset-level terms and the
  limitation.

## 2. Manifest and downloader

The manifest is a small text file. No binary is committed. It records for every
dataset the URL, license, license URL, and a content pin. Every case records the
exact input reference and its SHA-256.

`benchmarks/multimodal_data.py`:

- downloads to a `.part` file with a byte cap equal to the declared size; an
  upstream that grew is rejected before the disk fills;
- verifies size and SHA-256 before an atomic rename;
- reuses a cached file only when its size and SHA-256 match;
- replaces a corrupt cached file;
- scans a tar archive at most once for all requested members, with a member cap
  and a member-byte cap;
- verifies the GSM8K row SHA-256 and the gold answer;
- verifies the SpeechCoco caption and requires the pinned revision inside both
  asset URLs;
- decodes FLAC with `flac`, falling back to `ffmpeg`, and rebuilds the canonical
  PCM16/mono/16 kHz WAV. The decoded WAV SHA-256 must match the manifest;
- converts the FLEURS 32-bit float WAVs to PCM16 in Python with a fixed rule
  (scale by 32768, truncate, clamp), so the result does not depend on the
  ffmpeg build or its dithering. The member SHA-256 is checked before the
  conversion and the converted WAV SHA-256 after it. Members are matched by
  exact name and read into memory, so no archive path is ever written to disk.

Default cache: `~/.cache/umminfer-multimodal-benchmark`
(`UMMINFER_BENCH_CACHE` overrides it). Verify all inputs without a server:

```sh
python3 -I tests/multimodal_benchmark.py --verify-data
```

Observed on 2026-11-19: `PASS data: 70 cases verified (manifest 2026-11-19.1)`.

## 3. Runner

```sh
python3 -I tests/multimodal_benchmark.py \
  --base-url http://127.0.0.1:43200 --model qwen3-unified \
  --evidence-dir /tmp/mm-evidence \
  --accept-baseline              # first full run only
python3 -I tests/multimodal_benchmark.py \
  --base-url http://127.0.0.1:43200 --model qwen3-unified \
  --evidence-dir /tmp/mm-evidence-gate \
  --baseline /tmp/mm-evidence/baseline.json --rounds 3
```

Mandarin ASR only, and the full baseline once a service is up:

```sh
python3 -I tests/multimodal_benchmark.py --verify-data --tags zh
python3 -I tests/multimodal_benchmark.py \
  --base-url http://127.0.0.1:43200 --model qwen3-unified \
  --evidence-dir /tmp/mm-evidence-zh --tags zh --rounds 1
python3 -I tests/multimodal_benchmark.py \
  --base-url http://127.0.0.1:43200 --model qwen3-unified \
  --evidence-dir /tmp/mm-evidence --accept-baseline
```

The `--tags zh` run is a subset: it reports `subset=true` and can never
produce a stability `PASS` or establish a baseline. The third command
replaces the previous 54-case baseline, because the manifest version and
the case set changed.

Public endpoints only:

- `POST /v1/chat/completions` with `stream=true` for GSM8K, VQA, joint, and
  optionally ASR (`--asr-route chat`);
- `POST /v1/audio/transcriptions` for ASR by default;
- `GET /health` as a preflight signal. A failed health check is logged; each
  case is still attempted and classified.

Default execution is the full 70-case manifest. A subset requires `--cases` or
`--tags`. The report then has `subset=true`, the selected count, the manifest
count, and `excluded_case_ids`. There is no limit flag and no silent truncation.
A subset round is ineligible for a stability PASS.

Per case the runner saves:

- the exact SSE data payloads in `events/<id>.jsonl`;
- the final answer and the ASR transcript;
- HTTP status, `finish_reason`, `usage`, event count, and timings, including
  first-delta TTFT (`first_delta_ms`) and unified-terminal time
  (`terminal_ms`);
- the unified terminal kind/stage and `terminal_legs`. Exactly one `complete`
  terminal is required. The array remains `null` when no optional output legs
  were requested; when present, every reported leg must be `completed`;
- the input SHA-256 and a request summary with payload bytes replaced by length
  and SHA-256;
- the metric value and every failure assigned to the case.

Caps: 200,000 SSE events and 64 MiB of event text per case; 8 MiB per response;
64 KiB per run log. The run log is trimmed while streaming, so it never exceeds
the cap on disk. The case loop, the round loop (maximum 30), the download loops,
the tar scan, the GSM8K line index, and the SSE loops all have explicit bounds.

### Metrics

- **WER:** word Levenshtein distance divided by reference word count, after
  uppercasing and removing punctuation. Aggregated as corpus WER
  (`sum(errors) / sum(reference words)`).
- **CER:** character edit distance divided by reference characters. Both
  sides are NFKC-normalized, lowercased, and stripped of whitespace,
  punctuation, symbols, and control characters. Mandarin uses CER as its case
  metric and also reports WER over `jieba 0.42.1` dictionary-only tokens
  (`jieba.cut(HMM=False)`). Characters are never counted as Mandarin words.
- **VQA accuracy:** per question `min(1, matches/3)` over the 10 provided
  answers, then averaged. Normalization is declared in section 1.
- **GSM8K accuracy:** the last `#### <integer>` marker must equal the gold
  integer. A missing marker scores 0.
- **Joint:** corpus ASR WER against the spoken caption, plus the fraction of
  declared caption keywords present in the answer.
- **Performance:** p50 and p95 total latency (nearest rank), and
  `tokens_per_s` p50/p95 with its source label. Success rate counts cases
  without an infrastructure or public-contract failure.

### Evidence layout

```text
<evidence>/
  run.log
  summary.json
  stability.json
  baseline.json            # only with --accept-baseline
  round-1/
    report.json            # machine-readable aggregates and verdict
    failures.json          # counts plus one entry per failure
    cases.jsonl            # one line per case
    events/<id>.jsonl      # raw SSE data payloads
    answers/<id>.txt
    transcripts/<id>.txt
    human_review/<id>.json # free-form outputs and ASR-chat auxiliary answers
```

### Combined real coverage audit

`tools/multimodal_coverage_audit.py` combines disjoint real evidence sets without
rerunning or editing their case rows. Each source must have `summary.json`,
`round-1/{report.json,cases.jsonl,events/}`, and a verified `SHA256SUMS`.

```sh
/path/to/bench-venv/bin/python -I tools/multimodal_coverage_audit.py \
  --source accepted_vqa_joint=/tmp/accepted-vqa-joint \
  --source standalone_asr=/tmp/standalone-asr \
  --source gsm8k=/tmp/gsm8k \
  --cache-dir ~/.cache/umminfer-multimodal-benchmark \
  --output /tmp/combined-coverage-audit.json
```

The auditor requires the pinned `jieba 0.42.1`. It independently rebuilds raw
SSE responses, validates unique completed terminals and optional legs,
recomputes per-case and aggregate metrics, resolves all pinned inputs, and
checks WAV/image contracts. The sources must be disjoint and their union must
equal all 70 manifest ids. Missing metric values stay `null`; numeric zeroes
remain measurements. Combining subsets establishes coverage only. The output
keeps `stability_verdict=BASELINE_REQUIRED` and cannot produce a stability PASS.

## 4. Failure classes

Infrastructure problems and semantic quality never share a class.

| Category | Examples | Severity | Automatic action |
| --- | --- | --- | --- |
| `infrastructure` | connection refused, timeout, HTTP 5xx, missing input bytes, server never healthy | P1 | restart or redeploy gate, only when the failure is verifiably infrastructural |
| `public_contract` | valid request rejected with 4xx, malformed SSE JSON, missing `[DONE]`, missing `finish_reason`, missing/duplicate unified terminal, malformed or non-completed terminal legs, missing `asr_final`, `asr_final` after a VLM delta | P0 or P1 | none; the owning agent fixes `serve`/worker code |
| `quality` | empty answer, empty transcript, WER above 0.50, wrong VQA answer, wrong GSM8K answer, joint sub-metric below its floor | P2 | none; the owning model or prompt owner decides |
| `performance` | a case above `--case-latency-cap-ms`, or a p95 regression in the gate | P2 | none; performance work is a code or capacity change |

The stability gate treats only infrastructure and public-contract P0/P1 as hard
round failures. Quality and performance enter through the declared tolerances.

## 5. Stability gate

Constants in `tests/multimodal_benchmark.py`:

- `REQUIRED_CONSECUTIVE = 3`
- `MAX_ROUNDS = 30`
- `P95_TOLERANCE = 0.10`
- `LATENCY_GATE = "median_v2"`; `MEDIAN_TOLERANCE = 0.10`. The nearest-rank
  p95/max at n≤16 is dominated by host scheduling, so blocking comparison uses
  per-task medians. The six-case joint task is observation-only; p50/p95 remain
  recorded for every task.
- `QUALITY_TOLERANCE = {asr_en_wer: 0.02, asr_en_cer: 0.02,
  asr_zh_cer: 0.02, asr_zh_wer: 0.02, vqa_accuracy: 0.05,
  gsm8k_accuracy: 0.05, joint_asr_wer: 0.05, joint_keyword_hit: 0.05}`
- ASR is reported per language (`asr_en`, `asr_zh`) so a Mandarin CER and an
  English WER never share one number.

A round passes only when all of these hold:

1. the round ran the full 70-case manifest;
2. infrastructure failure count is 0;
3. public-contract P0/P1 count is 0;
4. every quality metric is within its tolerance of the accepted baseline
   (lower is better for WER and joint ASR WER);
5. every blocking task's median latency is within ±10% of the baseline under
   `median_v2`; joint latency and all p95 values remain observations.

A run returns `PASS` only when three consecutive rounds pass. Two passing rounds
return `INSUFFICIENT_ROUNDS`. Any failed round returns `FAIL`. A subset with an
accepted baseline returns `INELIGIBLE_SUBSET`. A run without an accepted
baseline returns `BASELINE_REQUIRED` (including a bounded subset without one).

The first full run cannot claim stability. Use `--accept-baseline` to write
`baseline.json`. Baseline creation is refused when the round is a subset, has an
infrastructure failure, has a public-contract P0/P1, or is labelled
`contract-test`. A later run must use the same manifest version and the same
case ids; otherwise the baseline is rejected with `BASELINE_MISMATCH`.

## 6. Feedback, fix, canary, and re-release

```text
benchmark run
  -> classify failures (infrastructure | public_contract | quality | performance)
  -> minimal fix
       infrastructure: verify, then restart or redeploy (automatic gate allowed)
       contract/quality/performance: code or semantic change by the owning agent
  -> affected cases + full black-box contract suite
  -> candidate canary (subset, never a stability claim)
  -> full 3-round benchmark against the accepted baseline
  -> promote, or roll back to the previous accepted baseline
```

Rules:

- Only verifiable infrastructure actions may be automated: restart, redeploy,
  readiness wait, and the gate that decides whether to continue. The harness
  itself starts and stops nothing.
- Code and semantic fixes belong to the responsible agent. The benchmark
  harness never edits code, prompts, thresholds, or the manifest.
- A fix is invalid if it changes a tolerance, deletes or skips a case, changes a
  reference answer, or writes a fake response. The manifest version and case ids
  are part of the baseline identity, so such a change breaks the comparison
  instead of hiding a regression.
- A canary subset is useful for a fast signal. It cannot produce a stability
  PASS.
- Rollback returns to the last accepted baseline and its model or deployment.
  Re-running the full gate after rollback is required before a new promotion.

## 7. Local tests

```sh
python3 -I tests/multimodal_benchmark_test.py
python3 -I tests/multimodal_coverage_audit_test.py
python3 -I tests/multimodal_benchmark.py --verify-data
```

The suite covers: WER, CER, Chinese CER (identical, fully wrong, two
substitutions, punctuation, empty reference), VQA accuracy, GSM8K extraction,
keyword hit rate; manifest structure, coverage and pins, including the 16
Mandarin cases; hash-mismatch rejection in the downloader; download
verification, cache reuse, corrupt-cache replacement, oversize rejection, tar
extraction; public-client
SSE and transcription parsing; failure classification for all four categories;
three-round PASS, insufficient rounds, infrastructure and contract failures,
p95 outside ±10%, quality drop beyond tolerance, subset ineligibility, baseline
identity; explicit subset reporting; the 64 KiB log cap; and a local
`contract-test` end-to-end run that produces the full evidence layout. The local
server is a fixture and is never presented as model evidence. The separate
coverage-auditor test checks independent metric arithmetic, missing-versus-zero
handling, raw SSE/terminal validation, and corrupt or unsafe checksum rejection.
