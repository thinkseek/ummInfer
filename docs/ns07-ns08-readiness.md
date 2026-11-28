# Task #85 NS-07/NS-08 defect correction and zero-control promotion gate

## Outcome: PROMOTED_ISOLATED_ENGINEERING_ONLY

Both retained Task #80 defects and the additional native-outbox race exposed by
Task #85 are fixed. The complete corrected six-way Before and After arms pass
all isolated hard gates, and the zero-control Web/macOS defaults are promoted.
The unified scorecard's `scoreboard_update_allowed=false` is preserved because
other real hardware/human gates are not all complete; North-Star scores are
unchanged. **Resident five-GPU production NS-07 and full real-macOS/human NS-08
are NOT_RUN.**

## Full frozen six-way Before / After

Both arms used the same physical GPU 0, native binary/model/config identities,
seed `70080`, frozen media, warmups/exclusions, six persistent lanes, real fault
injection, and corrected harness `bcba9754…4f8dd65`. Every measured lane ran
for the full 1,800 seconds. Raw evidence includes every attempt and expected
fault; no request, timeout, or outlier was dropped.

| Metric | Before: parent | After: candidate | After − Before |
| --- | ---: | ---: | ---: |
| New-work window (s) | 1800.000750020 | 1800.000702545 | −0.000047475 |
| Measured requests (3 warmups excluded) | 1998 | 1999 | +1 |
| Completed | 1627 | 1628 | +1 |
| Cancelled | 307 | 307 | 0 |
| Expected fault records (including cancellations) | 371 | 371 | 0 |
| Unexpected errors / terminal-free timeouts | **0 / 0** | **0 / 0** | 0 |
| Process crashes | 0 | 0 | 0 |
| Cross-session / stale / duplicate terminals | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| PCM mismatches / timed ACK mismatches | 0 / 0 | 0 / 0 | 0 |
| Recovery successes / attempts | **371/371** | **371/371** | 0 |
| Final `in_flight` | 0 | 0 | 0 |
| Same-residency device growth (MiB) | 0 | 0 | 0 |
| Same-process RSS growth (MiB) | 48.8046875 | 46.55078125 | −2.25390625 |
| Nearest-rank p95 latency (ms) | 5714.449327 | 5704.470684 | −9.978643 (−0.174621%) |
| Output-stage overlap / lease violations | 0 / 0 | 0 / 0 | 0 |
| Native-drain events / nonempty waits | 34 / 2 | 33 / 3 | — |

Every normalized hard gate is true in both arms, including complete lane and
modality coverage, delayed reads, exact native/public PCM, timed playback after
virtual consumption, isolated ports, exact provenance, owned-process exit, GPU
exclusivity, bounded memory, and complete manifests. Raw failures arrays are
empty in both accepted arms; the 371 expected faults per arm remain in raw
records and are not reclassified as unexpected successes.

Raw self-excluded manifests:

- Before: 6,868 entries, SHA-256
  `a21b27c7a62650423c6b708e46bc99859b1e9d43edc239baa6f445a80fdd79a9`;
  final verification SHA-256
  `4e7813b270854d872144c489522ebf410786fe71e81a0894cf6a2a5e6d0a263d`.
- After: 6,862 entries, SHA-256
  `43fbcbe28bc90236f3f28712e526d66cef5f03c2ba5e8a82c09233df9ab87371`;
  final verification SHA-256
  `8b14b2172d70cf30a071671fbbc8ebbfd79f8f69a9ac6d005f2531021c6444c9`.

These are isolated-candidate results, not the untouched resident production
service and not evidence of physical playback or perceptual quality.

## Paired real Chromium NS-08 and zero-control defaults

The default rendered DOM was measured without presets or Advanced interaction:

| Measure | Before `b23f6fd1` | After `9e2e5f6` |
| --- | ---: | ---: |
| Default-visible developer controls | 2 | **0** |
| Chromium checks passed / failed | 9 / 1 | **10 / 0** |
| Contract failures | 0 | 0 |
| Export/import, reconnect, source revision, cancellation/retry | 1/1/1/1 | 1/1/1/1 |

Before/After browser-arm manifests have 5/6 entries and SHA-256
`596774448bfd3aaf88eed1981836e9ac37eca253b416eb082be7c64b72d12fcf` /
`2254cbb86113d6fb00ff24c350b1eb42e4da64bc27ec618d166186f0d54b9016`.
Their fresh-process verification reports both PASS with SHA-256
`c53b533b1b019eff05c53f986b41fd18b28380589064b75ac7a1ae2fcb2ea07f` /
`4f2cd7f8618328f775c37ece11cbc5130d3c998ccd13e060a284611642acbbff`.
Task #76/#77 exact references, media, lossless replay, and lifecycle contracts
remain intact. Web and macOS developer controls stay available only behind
collapsed Advanced UI; no default journey requires them.

**Real Swift compilation/XCTest, real macOS execution, physical acoustics,
perceptual quality, and human walkthrough are NOT_RUN.** Linux source tests are
not a full NS-08 PASS.

## Unified evidence and safe-upgrade boundary

- [Paired eight-case scorecard](evidence/ns07-ns08-scorecard.json), SHA-256
  `0e3d85c8cf6b03a20fcac4c224dd10522aab949017ae4daf3a6177a6c54791ac`.
- [Independent scorecard verification](evidence/ns07-ns08-scorecard-verification.json),
  SHA-256 `ac736d62e24ccdc1f94d869af954c420331bd44d15da8593616571a7e5e5d32c`.
- Scorecard manifest: 16 files, SHA-256
  `4e789e76fd0f824c1f70da1343b432137f788edf2b89276f9ea1036a6eee266f`;
  independent verifier SHA-256
  `23a90ffc522502f5fb9256e617b98abe84328892a59dd5a84ca3869f1a93223b`.
- [Root-cause record](evidence/ns07-task85-root-cause.json), SHA-256
  `271811cbdcc7f4fb0c3886f16376fa292760cbd88622b40cf78343fd6a19b6e8`.

The [candidate manifest](../benchmarks/north-star/ns07-upgrade-candidate.json)
has SHA-256 `c0ce785707e8c35d7f2c810f6ba2890c744bddfec81999055686f27855619de4`.
Its [offline preflight](evidence/ns07-upgrade-preflight.json) is **PASS** with
zero failed checks, SHA-256
`279436ed8afa46d4976fc42167270eb77afe1083bc6cec16d81f346f143736a7`, and zero
network/process actions. Rollback is bound to exact parent `b23f6fd1`; resident
upgrade and rollback remain NOT_RUN and require explicit owner authorization.
No deploy file, owner script, resident process, or resident port was touched.

## Verification and cleanup

`umm-test-local`; remote NS-07 18/18 including MessagePack; Web, storyboard,
SSE replay, public PCM, realtime, client media/CLI, narrative/I2V, Showcase,
scorecard, and fail-closed upgrade tests pass. Feature-map validation covers
235 rows / 1,219 references and relative documentation links pass. AOCI is
hand-maintained with stable tracked files only; no unavailable CLI is claimed.

All task-owned engine, stage, tap, browser, and verifier processes stopped.
Final GPU 0 postflight is **3 MiB, 0%, zero compute processes**. No training or
optimizer work occurred.

The evaluated lineage starts at official main
`b23f6fd1f2b3c50165ddcc5cac86dc5aa7723dc4`. Candidate client defaults and the
first soak corrections are in `fdf8557e74311334f9510199411300a945033395`;
the final native-terminal drain correction is
`9e2e5f66dde8814e3ae8e286c57c358956326aab`. The unified scorecard executes at
`480f97b47d5fcf351c22fd9d4d30365d186823f1`, which changes only the stale NS-08
parent-test identity and does not alter either evaluated arm.

## Retained-defect isolation

The [compact root-cause record](evidence/ns07-task85-root-cause.json) binds the
following conclusions to retained raw SSE, native MessagePack taps, process
lifetimes, WebSocket frames, and exact monotonic timestamps:

1. **Output-stage starvation and trace rotation are harness behavior.** A voice
   lane could repeatedly reacquire TTS ahead of a waiting narrative Wan switch,
   and the TTS handoff watcher could lose its `outputs_started` row at the
   64-KiB UMM_TRACE rotation boundary. FIFO acquisition and rotation-persistent
   expected-leg tracking fix those paths without changing the 600-second
   serving deadline.
2. **A second full arm exposed a distinct native-outbox shutdown race.** Both
   retained requests had image, video, and every PCM chunk, but the harness
   stopped TTS 101.230533 ms / 79.676031 ms after the final forwarded PCM chunk
   and before the native success terminal. The only retained terminal for each
   final child was the later abort. A stage switch now waits for every accepted
   native TTS id to have a forwarded complete terminal, and fails closed if that
   drain does not occur within the existing request-timeout bound.
3. **The playback-ACK mismatch is a verifier clock-order defect.** The server
   accepted the retained ACK 165,456 ns after timed playback, but the client
   logged TX 55,446 ns after that acceptance because logging follows
   `socket.sendall()`. Exact ACK identity/finality is now ordered against the
   producer pre-send boundary, not the later TX log. PCM, issued-cursor, full
   digital-duration, monotonic-cursor, and delayed-read checks are unchanged.

No serving C++/CUDA runtime or public/client protocol schema change was needed.
Every deterministic regression failed before its fix and passed afterward,
including negative controls for missing native terminals and acceptance before
the real send boundary. The failed environment and full-duration attempts are
retained in [the negative-attempt registry](evidence/ns07-six-way-negative-attempts.json).

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
