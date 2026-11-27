# Task #80 NS-07/NS-08 production-resilience and zero-control readiness

## Outcome: NOT_PROMOTED

The candidate zero-control client defaults produced a measurable isolated Web
engineering improvement, but the real paired six-way soak failed in **both** full
arms. Under the preregistered promotion rule, those defaults were removed from
the final tree. The evaluation harness, frozen configurations, compact evidence,
and independent verifiers remain.

- Official main stayed `91310b36184c86a1c147debb27db8643cc0cb3e4` at the pre-gate
  fetch; no newer merge was available.
- Required accepted starting tree:
  `b4dca5b0e3f4fb18ecd857847cf2205c660cb8a7`.
- Evaluated candidate:
  `0c8fac8c1d3040fcbf62fe886a5a3686adaf16bf`.
- Candidate client reversion: `867567c` (full identity is in Git history).
- Final diagnostic-harness correction: `199445dd78a6976b81429cbcfd29bc7f84c3868c`.
- **Resident five-GPU production NS-07: NOT_RUN.** No deployment, restart, or
  owner-protected configuration change occurred.
- **Real macOS and human NS-08: NOT_RUN.** No Web-only full-PASS claim is made.
- North-Star scores are unchanged; `scoreboard_update_allowed=false`.

## Unified machine evidence

- [Paired NS-01..NS-08 scorecard](evidence/ns07-ns08-scorecard.json), SHA-256
  `929611300d5fb16cf844fb4d7cb64ec04b3b108693918fd918443b86bf1d609a`.
- [Independent scorecard verification](evidence/ns07-ns08-scorecard-verification.json),
  SHA-256 `6ab79cc75db6a3efcc1a89b2c09d9685708371a9abdceade6b3179d18ccd160f`.
- [All failed/short/interrupted attempts](evidence/ns07-six-way-negative-attempts.json),
  SHA-256 `ff7e40cc7df43b3ee12ffd9fc66afbc54ca66ad2340a57972908720ee3e9e1c1`.
- Final scorecard external manifest: 16 files, SHA-256
  `d44fb9e118fa9d99977d931886ee4dda83651db8cd24d4bfb5dbc9e0395ebdec`.
- Independent scorecard verifier SHA-256:
  `23a90ffc522502f5fb9256e617b98abe84328892a59dd5a84ca3869f1a93223b`.

The entry point is `tools/umminfer_ns_scorecard.py`; the frozen registry is
`benchmarks/north-star/cases.json`. Every NS-01…NS-08 case is present. Cases
without their required real hardware/deployment/human conditions remain
`NOT_RUN`/`NOT_COVERED`; offline contract PASS is never substituted.

## Full six-way Before / After

Both arms used the same physical H20 SM90 GPU 0, native binaries, exact model
file trees, frozen natural-footage/video and 16 kHz PCM16 speech inputs, seed
`70080`, two ephemeral loopback listeners, and the same harness/preregistration.
ASR+VLM were resident; FLUX/TTS/Wan were bounded to one leased output stage.
Six persistent lanes covered text/code, interleaved image/I2V/PCM, full-duplex
voice with barge-in/playback ACK, real natural-video input, seeded abort/retry,
and malformed audio/dimension recovery.

| Metric | Before: parent | After: candidate | After − Before |
| --- | ---: | ---: | ---: |
| New-work window (s) | 1800.000690572 | 1800.000769254 | +0.000078682 |
| Measured requests (3 warmups excluded) | 2211 | 2290 | +79 |
| Completed | 1876 | 1945 | +69 |
| Cancelled | 326 | 336 | +10 |
| Expected faults | 334 | 344 | +10 |
| Unexpected errors | **1** | **1** | 0 |
| Process crashes | 0 | 0 | 0 |
| Cross-session / stale / duplicate terminals | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| PCM mismatches | 0 | 0 | 0 |
| Recovery successes / attempts | **334/334** | **344/344** | +10/+10 |
| Final `in_flight` | 0 | 0 | 0 |
| Same-residency device growth (MiB) | 0 | 0 | 0 |
| Same-process RSS growth (MiB) | 45.3125 | 46.7421875 | +1.4296875 |
| Nearest-rank p95 latency (ms), all measured outcomes | 5628.182504 | 5483.147751 | −145.034753 (−2.57694%) |
| Stable samples with excess GPU processes | 0 | 0 | 0 |
| Output-stage lease/lifetime violations | 0 / 0 | 0 / 0 | 0 |
| Timed playback-ACK invariant | PASS | **FAIL** | regression present |

The p95 difference is diagnostic only: it does not offset either failure and is
not a promoted performance gain. Each arm had one real narrative request reach
the existing 600-second `stream timed out` boundary and end with `[DONE]` but no
unified terminal. After additionally retained one raw playback-ACK acceptance
mismatch. No failure was dropped and no duration/client count was reduced.

Before had 262 realtime response records and exact native/public joins for 15
HTTP plus 23 realtime legs. After had 281 realtime response records and exact
joins for 15 HTTP plus 305 realtime legs. The additional After ACK verifier
failure remains separate from byte-level PCM equality. Physical acoustics and
perceptual quality are NOT_RUN.

Raw manifests:

- Before: 8,032 files, SHA-256
  `adc929802d555bfe7b60ccdff55c113f2e7178293b54397fc8edaf4fc06278cb`;
  final verification SHA-256
  `19a42113d5c56c1f319f4ebbef7b771587beafc243be46a4f97150b517fd672d`.
- After: 8,323 files, SHA-256
  `4819cdafe347988631279f3a0af119407f85b02ec933c455ca90abcf8ca73f18`;
  final verification SHA-256
  `cb4750651c31e98446310bc0aa0acb0ad73a8259fc59eef7d0ea3ceead912da8`.

These full runs are isolated-candidate evidence, not the resident production
NS-07 gate and not evidence of zero leaks beyond their measured boundaries.

## NS-08 paired zero-control diagnostic

The evaluated candidate moved every modality/routing/sampling/frame/voice-output
and echo developer control inside collapsed Advanced UI, and aligned fresh macOS
narrative/voice defaults. Real Chromium measured the default rendered DOM with
no preset or Advanced interaction:

| Measure | Before `b4dca5b0` | After `0c8fac8c` |
| --- | ---: | ---: |
| Default-visible developer controls | 2 | **0** |
| Browser failures | 1 (zero-control baseline) | **0** |
| Contract failures | 0 | 0 |
| Exact export/import, reconnect, source revision, cancellation/retry contracts | 1/1/1/1 | 1/1/1/1 |

Before/After arm manifests contain 5/6 files and have SHA-256
`cf7ec471580885dd93b774210156bf13ab5da78b55b0554829f83d4a28e5d147` /
`afba8076e814cf5d848757637636bf33fd036e21f18129161da0ef082fdf8cca`.
Both arms passed `tests/ns08_zero_control_verify.py` in fresh processes;
verifier SHA-256 is
`e7aac1ea7c3e9bc040242c0997078b4c7acfe26049bd4ab2d927815a74363bea`.

This is a real isolated NS-08 engineering improvement only. It did not overcome
the NS-07 failures, so the candidate defaults are **not present in the final
tree**. Real Swift compilation/XCTest and macOS/human walkthrough were NOT_RUN;
Linux source tests are not a substitute. Existing accepted Task #77 lossless
Web behavior is preserved.

## Candidate preflight and rollback artifacts

[`ns07-upgrade-candidate.json`](../benchmarks/north-star/ns07-upgrade-candidate.json)
pins exact Before/After commits, source tree, Web/Swift sources, five native
binary hashes, five model-tree hashes, model config hashes, health invariants,
and abort conditions. Its SHA-256 is
`d0412f902e367640b74c255faed1837bb0e93546ddb6baed07bedbc0cbc8c22b`.

The [offline preflight record](evidence/ns07-upgrade-preflight.json) is **FAIL**,
as required for the non-promoted candidate, with SHA-256
`9cec3e165bb1f27c1d0be40c9ffa890c842ff9195ffddea04fc83333a3e460e1c1`.
It performed zero network/process actions. Resident upgrade and rollback are
both NOT_RUN; owner authorization is explicitly required. The final source
reversion is not represented as a resident rollback.

## Verification and cleanup

- `umm-test-local`: PASS.
- Existing Web 24 checks, storyboard verifier 10/10, SSE replay 10/10,
  public PCM 21/21, Showcase 28/28, narrative/I2V/geometry self-checks: PASS.
- Client media 13/13 and CLI parse/serve/URL groups: PASS; real MLX/ASR local
  regression remained skipped.
- Unified scorecard tests, NS-08 adapter/arm tests, upgrade fail-closed tests:
  PASS.
- NS-07 deterministic suite: 13 tests, 12 PASS + one local `msgpack` skip;
  the remote CUDA-capable environment ran all 13/13 including MessagePack.
- `tools/check_feature_map.py` and documentation links are revalidated during
  closeout. AOCI is hand-maintained; no unavailable CLI claim is made.
- All task-owned engine/browser processes were stopped. Final postflight on
  physical GPU 0 was **3 MiB, 0%, zero compute processes** after the full arms.
  GPUs 1–7 and the resident service were not used or contacted by the soak.

The required owner decisions for any future production rollout, real macOS
walkthrough, physical acoustic tests, and human perceptual review remain open.
