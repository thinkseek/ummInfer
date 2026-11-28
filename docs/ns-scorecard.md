# Unified North-Star scorecard and evidence boundary

Task #85's corrected full-pair result, retained-defect root causes, negative
attempts, and isolated-engineering-only promotion are recorded in
[ns07-ns08-readiness.md](ns07-ns08-readiness.md). Task #80's historical failed
arms remain preserved there and in the negative-attempt registry.

`tools/umminfer_ns_scorecard.py` is the single NS-01…NS-08 aggregation entry
point. Its frozen case registry is
[`benchmarks/north-star/cases.json`](../benchmarks/north-star/cases.json). The
runner invokes the existing case-specific contract drivers in registry order;
it does not reimplement their parsers, media checks, or model evidence.

## Offline contract run

```sh
python3 -I tools/umminfer_ns_scorecard.py \
  --evidence-dir /caller/selected/new-empty-evidence-directory
```

The evidence directory is required, must be external to the repository, and
must be absent or empty. Each case writes bounded command logs beneath that
external root. The resulting `ns-scorecard.json` always contains exactly eight
ordered cases, including explicit `NOT_RUN`/`NOT_COVERED` states when hardware,
real models, macOS, or human review are unavailable. Contract PASS is not a
North-Star hard-gate PASS.

The runner records the exact execution commit, registry and harness SHA-256,
each driver SHA-256 and exit status, raw log hashes, frozen thresholds, every
reported failure, and a complete external self-excluded `SHA256SUMS`.

## Paired Before / After arms

A real case adapter writes `NS-07.json` or `NS-08.json` to a caller-owned arm
directory. Supply the accepted parent and candidate independently:

```sh
python3 -I tools/umminfer_ns_scorecard.py \
  --before-arm /caller/selected/parent-arm \
  --after-arm /caller/selected/candidate-arm \
  --evidence-dir /caller/selected/paired-evidence \
  --require-clean
```

An arm report carries only compact aggregates, all failures, hard-gate states,
source/binary/model/config identities, raw-evidence hashes, and external file
counts. Raw SSE/WS frames, PCM, images, videos, screenshots, model files, and
process logs stay outside Git. The runner copies the normalized arm JSON into
the sealed evidence root so every `result_sha256` is independently checkable.

- NS-07 paired engineering comparison requires the full 1,800-second, six-lane
  schedule, zero unexpected errors/crashes/stale or cross-session events,
  duplicate terminals, or PCM mismatches; every injected fault must recover,
  `in_flight` must end at zero, resource growth must stay within the frozen
  same-residency bounds, and p95 latency must not regress by more than 5%.
- NS-08 paired engineering comparison requires a real Chromium default-DOM
  probe, zero candidate-visible developer controls, zero contract/browser
  failures, and preserved source-bound revision, cancellation/retry,
  export/import, and reconnect behavior.
- Neither result is a resident five-GPU production gate, real macOS execution,
  physical-acoustic validation, perceptual-quality assessment, or human
  walkthrough.

## Preregistered isolated NS-07 soak

The six-lane schedule, 1,800-second minimum, physical-GPU0 boundary, lease rules,
resource methods, and hard gates are frozen in
[`ns07-six-way-preregistration.json`](../benchmarks/north-star/ns07-six-way-preregistration.json).
All executable, model, media, and output locations are caller-selected:

```sh
UMMINFER_LIVE_OPT_IN=1 python3 -I tests/ns07_six_way_soak.py \
  --evidence-dir /caller/selected/new-empty-soak-evidence \
  --source /caller/selected/exact-source-checkout \
  --models asr=/caller/selected/asr-model \
  --models vlm=/caller/selected/vlm-model \
  --models flux=/caller/selected/flux-model \
  --models tts=/caller/selected/tts-model \
  --models wan=/caller/selected/wan-model \
  --server /caller/selected/umm-infer \
  --vlm /caller/selected/umm-native-stage-qwen38 \
  --asr /caller/selected/umm-native-stage-asr \
  --tts /caller/selected/umm-native-stage-tts \
  --wan /caller/selected/umm-native-stage-video \
  --voice-wav /caller/selected/voice-16k-pcm16.wav \
  --barge-in-wav /caller/selected/barge-in-16k-pcm16.wav \
  --natural-video /caller/selected/natural-input.mp4 \
  --python /caller/selected/python --library-path /caller/selected/lib \
  --execution-commit EXACT_SOURCE_COMMIT --arm after --duration-s 1800
```

The driver allocates two distinct ephemeral loopback ports and refuses the
resident port. It starts/stops only task-owned processes, retains uncapped raw
evidence, runs the independent verifier initially and again over the complete
self-excluded manifest, and writes normalized `NS-07.json`. A short diagnostic
is explicitly unable to produce `ISOLATED_PASS`. The five-GPU resident gate is
always `NOT_RUN` in this task's arm.

For the offline safe-upgrade record, `tools/umminfer_ns_upgrade_check.py`
cross-checks a candidate manifest against the normalized After arm, including
exact identities, health invariants, abort conditions, and dry-run rollback:

```sh
python3 -I tools/umminfer_ns_upgrade_check.py \
  --candidate /caller/selected/candidate.json \
  --after-arm /caller/selected/soak-evidence/NS-07.json \
  --out /caller/selected/new-preflight-report.json
```

It performs no network/process action and cannot authorize a rollout.

## Independent fresh-process verification

Produce and independently verify each NS-08 browser arm with caller-selected
source, evidence, Chromium, and accepted Playwright-core paths:

```sh
python3 -I tests/ns08_zero_control.py \
  --source-root /caller/selected/source \
  --evidence-dir /caller/selected/arm-evidence \
  --arm after --source-commit EXACT_COMMIT --browser \
  --chromium /caller/selected/chrome \
  --playwright-core /caller/selected/playwright-core
python3 -I tests/ns08_zero_control_verify.py \
  --arm-dir /caller/selected/arm-evidence \
  --out /caller/selected/arm-verification.json
```

The browser probe performs no generation or Advanced interaction. It records
startup model discovery separately from keyboard-action traffic and uses actual
rendered visibility, not merely layout rectangles, for closed disclosures.


```sh
python3 -I tests/ns_scorecard_verify.py \
  --scorecard /caller/selected/paired-evidence/ns-scorecard.json \
  --manifest /caller/selected/paired-evidence/SHA256SUMS \
  --root /caller/selected/paired-evidence \
  --out /caller/selected/outside-the-sealed-root/verification.json
```

The stdlib-only verifier does not import the producer. It checks complete case
coverage, sealed arm identities, paired-claim consistency, exact self-excluded
manifest coverage, and every file hash. A missing, extra, or modified file fails
closed. The verification report is normally written outside the sealed root so
its own creation cannot invalidate the manifest it just checked.

## Focused regression commands

```sh
python3 -I tests/ns_scorecard_test.py
python3 -I tests/ns08_zero_control_test.py
python3 -I tests/macos_zero_control_test.py
python3 -I tests/ns_upgrade_check_test.py
python3 -I tests/web_ui_contract_test.py
```

The Swift package still requires actual macOS for compilation and XCTest. The
Linux source checks are useful regression evidence only and never authorize an
NS-08 full PASS.
