# Qwen3.8 temporal motion-run selection — Task #61

## Decision and claim boundary

The preregistered `qwen38_temporal_motion_run_v1` candidate was **accepted for the
explicit `policy="temporal"` path only**. It coalesces each contiguous
above-threshold motion run to one representative: greatest motion, then greatest
scene score, then earliest decode index. The existing scene-run pass, global
priority, first/last anchors, deterministic uniform fill, chronological output,
source PTS, temporal-2 packing, mRoPE, and 4/8/16 budgets remain in force.

Default video input and every explicit uniform arm are unchanged. The candidate
does not become the default policy and does not change the request schema or any
threshold.

Bounded status after the SM90 run on 2026-11-27:

- `REAL_GENERATION: MEASURED_LIMITED_FIXTURE`
- `LIVE_LATENCY: CONTROLLED_LIVE`
- `FIXTURE_SUBSTRING_ACCURACY: MEASURED_LIMITED_FIXTURE`
- `PERCEPTUAL_QUALITY: NOT_RUN`
- generalized quality claim: `false`

Four synthetic semantic cases, with deterministic repeated outputs, do not
establish generalized video understanding. In particular, the primary temporal-8
improvement has two improved and zero worsened case-mode pairs; the exact
two-sided sign-test p-value is **0.5**. This is a controlled fixture promotion,
not statistical or perceptual proof of a general improvement.

## Preregistration and single-factor boundary

`docs/evidence/video-temporal-motion-run-preregistration.json` was committed before
any runtime edit or candidate response. Its SHA-256 is
`f62f2f8aa3f11db61fdda6a496ce41fd505304ea6c26fe51ffe795d6a5d8bd41`.
It fixes the primary metric, no-regression rules, sequential GPU-1 schedule,
uncertainty treatment, and rejection rule. The frozen `temporal-clean-r1` bytes,
questions, expected terms, scoring, thresholds, and selection-miss controls were
not edited.

| Identity | Baseline | Candidate |
| --- | --- | --- |
| Source commit | `b5aac196d638f7c011509f698a4fe9b8653c3767` | `267bdeb34d5ac8c8117863191a62e72030c79b7d` |
| Source-state SHA-256 | `20c7a26baa9b18cfdffde21396f5db758f80a73a1ad9b0bdd7eb5e74339673b0` | `11033a6c5b66c9303880dfc1c54a95821c458b7f69ca12a6edb327435b280731` |
| Gateway SHA-256 | `a3462ae8662ba70459d3188f5ed4873609e55fcb79c689147d8f20aba96d7487` | same |
| Native stage SHA-256 | `121b68aaedfa641e728ec024aa5d12914bcfbf2ed2c4077048ea66778b1b6ce5` | `7fa97c7cf859ecad7014b062c490442382be888d77bd241dd50b6047922462a0` |

Both descend from official main
`49d34063b7ecb61d0571bf962b11f1a71c0d039f`. The baseline/candidate runtime diff
contains only `native/qwen38/video_sampling.cpp` and
`native/qwen38_tests/test_video_input.cpp`; its SHA-256 is
`8d3e2b3272671db3514c54875b43298055266387a0858225d643218a1ffdcbfd`.
Later finalization strengthened cleanup and hit-to-miss auditing without issuing
more requests or changing either recorded runtime.

The model is local Qwen3.8-27B `qwen3-native`, revision
`UNKNOWN_LOCAL_NO_GIT`, with 29 hashed files. `config.json` SHA-256 is
`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`; the
canonical model-map SHA-256 is
`c37ba5f6860f94ef5aeab25bfb513b0c0c6d8f18b331e83090e7601995e78009`.
Frozen manifest and original preregistration SHA-256 values remain
`62f475a74732fa3504850ebe4517dc98903ba39202dce457441c52133a6c96c0` and
`96608dc793d54f64f55a7c46862d1b1737cbf807a8807d1b22b2fcd0502d2a77`.

## Controlled schedule and technical gates

Each phase ran a fresh serial stack and then a fresh batch-four stack on the same
explicitly idle **physical GPU 1**. They were sequential, not simultaneously
resident. GPUs 0 and 7, GPUs 2–6, and the resident service were not used. Prefix
caching was disabled, temperature was zero, top-p was one, seed was 510051, and
the limit was 64 output tokens.

Each phase contains 448 requests: 112 declared warmups and 336 measured requests.
Default runs at 8; uniform and temporal run at 4/8/16. Across both phases there
are **896 completed and native-valid requests**, with no failed or stale record.

- Baseline: 448/448 complete/native-valid and 400/400 parity checks.
- Candidate: 448/448 complete/native-valid and 400/400 parity checks.
- Parity covers default versus uniform, serial versus batch-four, and all
  warmup/measured repetitions; every request had `prefix_reused=0`.
- All 256 corresponding default/uniform cross-phase requests, raw answers,
  selected RGB/source/PTS sequences, temporal-2 layouts, and mRoPE diagnostics
  are identical.
- A fresh stdlib-only verifier independently decoded all 1,796 source frames,
  recomputed candidate luma/scene scores and both selectors, rebound every native
  selected RGB fingerprint and source PTS, and re-applied the frozen scoring
  without importing producer scoring.
- Every schedule step ended with `in_flight=0`; pre/post whole-device exclusivity
  and residual-memory checks passed.

The four selection-miss controls stay outside every quality denominator. Their
hit counts were unchanged between phases: default-8 0/24; uniform-4 3/24,
uniform-8 0/24, uniform-16 0/24; temporal-4 3/24, temporal-8 0/24,
temporal-16 3/24. Every raw control answer remains in the evidence.

## Primary quality result and uncertainty

The primary population is four quality-scored cases × three measured repetitions
× two modes = 24 answers per arm/budget. The frozen casefold substring rules are
unchanged. Wilson intervals below describe request-level proportions only;
repetitions are not independent semantic examples.

| Arm / budget | Baseline hits | Candidate hits | Baseline Wilson 95% | Candidate Wilson 95% | Improved / worsened case-mode pairs | Sign-test p |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| default / 8 | 18/24 | 18/24 | [0.5510, 0.8800] | [0.5510, 0.8800] | 0 / 0 | 1.0 |
| uniform / 4 | 6/24 | 6/24 | [0.1200, 0.4490] | same | 0 / 0 | 1.0 |
| uniform / 8 | 18/24 | 18/24 | [0.5510, 0.8800] | same | 0 / 0 | 1.0 |
| uniform / 16 | 18/24 | 18/24 | [0.5510, 0.8800] | same | 0 / 0 | 1.0 |
| temporal / 4 | 6/24 | 6/24 | [0.1200, 0.4490] | same | 0 / 0 | 1.0 |
| **temporal / 8** | **12/24** | **18/24** | **[0.3143, 0.6857]** | **[0.5510, 0.8800]** | **2 / 0** | **0.5** |
| temporal / 16 | 18/24 | 18/24 | [0.5510, 0.8800] | same | 0 / 0 | 1.0 |

There are zero temporal hit-to-miss repetition transitions at every budget. The
candidate temporal-8 therefore exceeds its baseline and matches, but does not
exceed, default/uniform-8.

The temporal-8 raw answers were identical across modes and repetitions:

| Case | Baseline | Candidate | Frozen result |
| --- | --- | --- | --- |
| `motion-first-en` | `rectangle` | `triangle` | miss → hit |
| `scene-cut-zh` | `三角形` | `三角形` | hit → hit |
| `near-static-ja` | `円` | `円` | miss → miss; no alias added |
| `bframe-first-es` | `cruz` | `cruz` | hit → hit |

Temporal-4 and temporal-16 raw quality answers are unchanged. This isolates the
observed primary change to the motion-clean case at budget 8.

## Latency and resources

The no-regression limit is candidate median ≤ 1.05 × baseline for both total and
native elapsed time in each temporal mode/budget. Each median aggregates 24
measured requests. Batch-four includes concurrency-four queueing and is not a
serial speedup comparison.

| Mode / budget | Baseline total / native ms | Candidate total / native ms | Native ratio | Result |
| --- | --- | --- | ---: | --- |
| serial / 4 | 432.18 / 430.0 | 426.12 / 423.5 | 0.9849 | PASS |
| serial / 8 | 542.70 / 540.0 | 535.37 / 532.5 | 0.9861 | PASS |
| serial / 16 | 686.96 / 684.0 | 684.97 / 682.0 | 0.9971 | PASS |
| batch-four / 4 | 1002.22 / 997.5 | 842.20 / 840.5 | 0.8426 | PASS |
| batch-four / 8 | 1282.17 / 1278.5 | 1281.14 / 1278.0 | 0.9996 | PASS |
| batch-four / 16 | 1586.11 / 1583.0 | 1637.98 / 1634.5 | 1.0325 | PASS |

These are no-regression observations, not a latency-improvement claim. Raw
per-case p95 and dispersion remain in `report.json`.

| Phase / mode | NVML samples | Whole-device peak | Post-stop |
| --- | ---: | ---: | --- |
| baseline serial | 1342 | 55,653 MiB | 3 MiB, zero compute processes |
| baseline batch-four | 921 | 57,897 MiB | 3 MiB, zero compute processes |
| candidate serial | 1332 | 55,653 MiB | 3 MiB, zero compute processes |
| candidate batch-four | 917 | 57,897 MiB | 3 MiB, zero compute processes |

All 12 task-owned stack processes accepted SIGTERM; no SIGKILL was used. Only
task-owned jobs were stopped.

## Evidence, independent verification, and tests

The external sealed bundle is `/tmp/umm-task61-evidence`; its archive is
`/tmp/umm-task61-evidence-r1.tar`. It retains every raw answer, SSE frame list,
request, native diagnostic, timing, resource sample, provenance record, native
log, launch identity, cleanup record, candidate oracle, frozen input, and analysis
source.

| Artifact | SHA-256 / count |
| --- | --- |
| Root `SHA256SUMS` | 1,891 entries; `ab96aa11fda5215f43f80426f41de9ac96330a5fc2c625cedb037b89db5ff13a` |
| Nested frozen-fixture `SHA256SUMS` | 48 independently checked entries |
| `report.json` | `4c527b05ed2cd2f299c37cd261769131446a552dcaf85cb13eabb1cfe4026d86` |
| Evidence archive | `4416ad35ef569f5999f6f55a02ecb8bd86ccae4afdbc9b3a3ed7db2de8930980` |
| Independent verification receipt | `00930f306d0d0f47cfe62442f899ee5e5219203120a59b4ea6ab292bda42985b` |

Fresh local and remote verifier processes returned byte-identical PASS receipts.
Both verified the complete checksum file sets; the remote process used the pinned
FFmpeg toolchain. The sealed verifier source SHA-256 is
`40bf81e9a945f1ce3e54c10ee55f54558bf59371f02e14ccaaa0eeed8fc1fb25`.

```sh
python3 -I tests/temporal_motion_run_verify.py \
  /tmp/umm-task61-evidence \
  --output /tmp/umm-task61-verification-fresh.json
```

Final checks:

- `umm-test-local test`: PASS.
- `python3 -I tests/temporal_motion_run_test.py`: 6/6 PASS.
- `python3 -I tests/temporal_clean_test.py`: 17/17 PASS.
- `python3 -I tests/long_video_temporal_test.py`: 14/14 PASS.
- `python3 -I tests/controlled_measurement_test.py`: 38/38 PASS.
- `python3 -I tests/temporal_quality_r4_test.py`: 15/15 PASS.
- Clean, short-video, and long-video live-harness self-checks: PASS.
- Remote Release native CTest: 3/3 PASS, including Qwen host and video-input
  tests. An initial CTest invocation had not built the unrelated ASR test target;
  building it and rerunning gave 3/3 without source changes.
- Local native `qwen38_video_input_tests`: PASS.
- Local non-Release `qwen38_host_tests`: the existing fast-nucleus timing
  assertion remains **FAIL** (8.797 ms/token after optimization versus its 2 ms
  threshold); no assertion or implementation was weakened. The same Release host
  test passed remotely.

A machine-readable distillation is
`docs/evidence/video-temporal-motion-run-sm90.json`. Raw evidence, not this summary,
is authoritative for per-request observations.

## Remaining limits

No human/perceptual evaluation, external natural-video corpus, new semantic case,
or generalized quality benchmark ran. Those remain `NOT_RUN`/`NOT_COVERED`. The
small synthetic fixture cannot resolve statistical significance, and the Japanese
`円` miss remains unchanged. No realtime/TTS, online-training, client, deployment,
or resident-service behavior was changed or validated.
