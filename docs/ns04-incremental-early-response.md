# NS-04 stable-complete incremental-response candidate (Task #83)

## Result

**NOT_PROMOTED; the candidate is absent from final runtime.** One bounded design
started the existing native VLM/streaming-TTS path only after three consecutive,
byte-identical completed ASR previews exactly matched a terminal command grammar.
It was correct on the frozen journey, but replay-inclusive first-barge→new-audio
remained about **2.718 s**, not strictly below **800 ms**. All three pairs also
retain `exact_pairing=false` for interrupted suffixes. No second candidate or
eligibility relaxation was attempted.

The final serving/native tree is identical to accepted Task #82. The NS-04 score
stays 80%; physical microphone/speaker, acoustic echo and perceptual quality remain
**NOT_RUN**. This is digital-loopback evidence only.

## Preregistration and the single candidate

[Preregistration v1](../benchmarks/ns04-incremental-early-response/preregistration.json)
was committed as `0767f8fa5d99e9b2bec2d4e017e514ec8121dfd2` before candidate runtime
edits or new measurements. Its SHA-256 is
`8883a36ade108db8c6bac14a1acb78584356531fa7c019cd8f4e7711cec8b9a9`.
The accepted parent is `ec1b9235e4ecd69340571d595726ed503bc42b47`; official main was
still `91310b36184c86a1c147debb27db8643cc0cb3e4` at preregistration.

Eligibility required all of the following:

1. Three completed previews from one segment, with identical trimmed UTF-8 text.
   Streaming fragments and confidence-like substring matches never qualified.
2. A 4–96-codepoint, single-line, valid UTF-8 command ending in terminal punctuation.
3. An exact full match to only `停一下[，,]?直接讲土星环[。！!？?]` or
   `换成英文讲[。！!？?]`. The submitted VLM text was the eligible ASR text itself.
4. No prefix, unresolved bracket, replacement character, negation, conflict,
   correction or open coordination. Uncertain/unseen input waited for ASR final.
5. Any later completed preview/final mismatch cancelled uncommitted work, dropped
   queued text/audio/ACK/terminal events, emitted a revision cancellation with
   `history_recorded=false`, and used the ordinary final-response path. An exact
   final retained one response rather than submitting a duplicate.

Execution commit `6263fb9593a98e9a9f6a5c3b0badb27fc912ad29` contained only that
candidate and its deterministic tests. Revert
`234635c7e31f96d52364b05b0c9f481636d8f47c` removed it after the gate failed.
The final raw-evidence harness commits are
`effe2eff21ef160b3b73f7aa80410102565c37f2` and
`d63f67c725f71aa5e59338219e8f27d4abc74702`; neither changes serving/native code.

## Frozen inputs and paired execution

The exact Task #82 v2 solar-system → Saturn-rings → English recordings, SHA-256
identities, 3,200-byte chunks, 100 ms pacing, six end-silence chunks, 400 ms VAD
silence, timed virtual playback, 768 VLM tokens, 8,192 TTS frames, Vivian voice,
models and sampling were retained. No transcript was injected. The six fresh runs
used order `before-r1 → after-r1 → before-r2 → after-r2 → before-r3 → after-r3` and
only explicitly idle physical GPU 7. The resident service and GPUs 0–6 were not used.

The retained Task #82 Before capture already showed that the Saturn preview became
complete only around 1.647 s after barge-in. Earlier text included `停一下，直接。`,
`停一下，直接讲土。` and `停一下，直接讲土星。`; none is a complete Saturn-ring
command. This observation was fixed in preregistration, not used to loosen the rule.

## Preview identity and incremental work

The independent verifier decodes raw WebSocket frames, native VLM/TTS MessagePack
submits and server diagnostic logs without importing the live producer. It recomputes
trimmed identities, stability counts and the grammar, binds the eligible digest to
one provisional VLM request, and requires exact final intent before accepting any
early public audio.

| After metric, ms | First interruption median / p95 / max | Second interruption median / p95 / max |
| --- | --- | --- |
| Barge → eligible | 2,054.840 / 2,055.083 / 2,055.110 | 1,445.788 / 1,445.841 / 1,445.847 |
| Barge → VLM submit | 2,055.071 / 2,055.347 / 2,055.377 | 1,446.076 / 1,446.080 / 1,446.081 |
| Barge → first TTS submit | 2,519.328 / 2,520.166 / 2,520.259 | 1,830.169 / 1,831.113 / 1,831.218 |
| Eligible → VLM submit | 0.242 / 0.265 / 0.268 | 0.229 / 0.287 / 0.293 |
| Eligible → first TTS submit | 464.591 / 465.093 / 465.149 | 384.450 / 385.332 / 385.430 |

The eligible Saturn SHA-256 is
`51cc302834f710c2764a9ab361e4929fadcdaaa7e1aa112c201aa8cf863df0b0`; English-switch
is `6aca4ab2637d30ac00d651114f2a69a09a1b9127e6c62d750958e270a2d8e18f`.
All After previews/finals matched those eligible identities, with zero preview
revisions and zero incorrect early audible responses. Before had no candidate work.
The rule's third-preview delay is not hidden: it is the main reason the candidate
cannot approach 800 ms on the frozen semantic input.

## Before/After timing result

All values are milliseconds. Quantiles use linear interpolation at `(n-1)*p`; every
sample and Tukey outlier remains in the compact scorecard. There were no Tukey
outliers in the primary full-path samples.

| First-interruption metric | Before median / p95 / max | After median / p95 / max | Result |
| --- | --- | --- | --- |
| Barge → cancel | 4.445 / 4.561 / 4.574 | 4.486 / 4.675 / 4.696 | PASS <300 |
| Cancel → digital silence | 1.511 / 1.533 / 1.535 | 1.476 / 1.498 / 1.501 | PASS <300 |
| Barge → speech stopped, replay included | 2,209.781 / 2,210.106 / 2,210.142 | 2,209.886 / 2,210.627 / 2,210.709 | unchanged diagnostic |
| Speech stopped → ASR final | 46.059 / 46.422 / 46.463 | 87.019 / 94.635 / 95.482 | diagnostic |
| ASR final → new audio | 622.444 / 622.999 / 623.061 | 421.504 / 421.517 / 421.518 | PASS <800 |
| **Replay-inclusive barge → new audio** | **2,878.645 / 2,878.751 / 2,878.763** | **2,717.846 / 2,718.103 / 2,718.132** | **NOT_MET <800** |

Second-interruption cancel median/p95/max was 1.732/1.892/1.910 ms Before and
1.824/1.894/1.902 ms After. Its full-path median was 2,097.618 ms Before and
2,018.759 ms After; these are diagnostics, not substitutes for the required first
full-path gate. Every accepted After first-interruption measurement exceeded 800 ms.

## Exact PCM, history and lifecycle evidence

Every raw capture independently proves the complete append schedule, exact played
PCM prefixes, final accepted playback cursor, pending-PCM clearing, two truncated
turns, exact canonical VLM history, three expected response identities, no stale or
cross-session events, no duplicate terminal callback kinds, no decreasing accepted
cursor, final `in_flight=0`, process exit and immediate following-turn recovery.
Both interruption ACK cursors remain 30 and 0; cursor zero is valid inside the first
TTS leg. The final English response is English-dominant, Saturn/ring topical and has
nonempty public PCM.

Final English text and PCM match in every pair, as do transcripts, played PCM,
accepted cursors and history prompts. However, all three pairs differ in the second
interrupted response's emitted/pending PCM and text suffix; pair 2 also differs in
the first interrupted text suffix. The scorecard therefore retains
`exact_pairing=false` and treats it as a failure rather than deleting or weakening
those checks. This independently prevents promotion.

## Deterministic and real controls

- The preregistered C++ state gate first failed to compile because the candidate was
  absent (red log SHA-256 `bb0a2d235f09dec9d927452029a690d258f654fb7007984972dd8e3f70ce11e3`),
  then passed eligibility, prefix revision, negation, language-switch,
  incomplete-clause, false-eligibility and retry fixtures (green log
  `37c7b84ec93229bb06179ed48fe509996e9651da3a9dc66dc1419301d20841b4`).
- The actual fake-WebSocket candidate test passed exact-final single ownership,
  preview-revision cancellation, stale suppression, no superseded history and
  immediate retry. Final Python fixtures retain the adversarial scorecard and
  independent eligibility checks after the runtime revert.
- Final `umm-test-local`: **PASS**, log SHA-256
  `e603338d158302724339b73d22fe82a5f88ea6ef16771d5e8a8f95de3a9f8f9a`.
- Final native TTS host CTest: **PASS**, log SHA-256
  `bd075de467f2f469268d98e0e7a9524a8daf9cb750d232b482514d97d76d8e1f`.
- Real native TTS CUDA determinism: **3/3 PASS**, log SHA-256
  `801284fbeefbe308f2e7c027fd587a007a83ad19a92574f55acefdcf7cc15fe7`.
- Public HTTP + delayed realtime: **PASS** with 25 HTTP responses (24 complete +
  one exact cancel/retry), 29/29 responses/final ACKs, 493 exact PCM legs, 4,176
  chunks, 115,034,880 bytes and exactly 4,125−29 = 4,096 pending audio events.
  Fresh verifier SHA-256 is
  `9640cdd5009670a5b09b9385fd399c0bd3086b7fafc04373796c3021b52ffbf4`; its
  1,251-file nested manifest is
  `0d4af98071e6406f022131fe649b0fd81b85447790e6fe464d22d453e858595f`.
- Realtime network/reconnect: **PASS** with six sessions, three rounds, 15
  connections, 41 connection-scoped identities, 30 exact next-turn prompts, 27
  terminal callbacks and zero stale/cross-session/duplicate/decreasing-cursor
  events. Verifier/manifest SHA-256:
  `b5d60657fad795455524297945e3166b0706bcfdaed4241b642d9bdcf4f7334f` /
  `5c2918aad79c694293a7d7ce7d9317d2c8ca069db439128878dc06eb0af1ed5e`.
- Task #81 cursor-10 NS-04: **PASS**, exact played PCM
  `ccda9cf40688346b658929d145f0fc03ff050b4f4ce7eda3201f94b90a37b003`, zero stale
  events and final `in_flight=0`; independent report
  `167c42a8b46997a7a0409903ce8fc3369f94083228117a08f69c2f6c5d59dac3`.
- Task #81 exact Chinese NS-01 raw lifecycle: **43/43 natural EOS PASS**, 43 exact
  native/public PCM legs, clean text `883802d8…b572`, 14,987,520-byte PCM
  `2ce2c534…4f78`; independent report
  `1083b6ff29f7a300b20e462a7b703a65b2669ed0da1a9de394f27e29937f7718`.
  TTFA/image/I2V and the full ideal NS-01 case remain NOT_MET/NOT_RUN.
- A final rebuilt, reverted runtime passed the complete journey and fresh raw
  verification at replay-inclusive 2,878.693 ms; gateway SHA-256 is
  `1bcc15f1842d7aa02fb44206631c5142b204a0e6f40113a9ba0710997db79721` and
  report SHA-256 is
  `0e2159012b910eb079b2e663de14c3281ccb2355b4a462f8a13ab7ae04ed142e`.
- Feature-map validation: **229 rows / 1,170 references PASS**, log SHA-256
  `6f91e2e6ada2f103bd7f4c4440ebedd4cfee6a7306420b8210933ea8dbf85585`.
- Markdown validation: **64 files / 277 local targets / 67 anchors PASS**, log
  SHA-256 `5654b1f93b89db04bc9f29a995128a5c37338cc24206471ed7c11b192b1a33a8`.

The compact machine scorecard is
[ns04-incremental-early-response-sm90.json](evidence/ns04-incremental-early-response-sm90.json),
SHA-256 `e0c709abe3ce9b8597e557adb7b9786a36d5983e1954f9f745182e2791ed4b76`.
Its external pre-projection scorecard identity is
`8fa2ab6fffac26ae4aa441d1d27c87745a57013731c775651af22afdb4a89269`.
The complete external self-excluded manifest covers **5,174 files**, passed two fresh
audits, and has SHA-256
`b0c16a1ab2dca1dff47320a01e81268de21f80bb3a42b7462a7fcdaa5984ddb8`.
Six additional fresh processes using only bundled verifier source and frozen input
bytes produced byte-identical independent reports. Raw media, wire events and per-leg
payloads stay outside Git in a caller-selected evidence directory; no absolute output
default was added.

## Retained harness negatives and boundaries

The first After launch failed before any request because the test-only build had not
installed the gateway executable. The empty attempt was retained separately and is
not a measured sample. The first After verifier also incorrectly counted the later,
intentional second barge as a revision of the Saturn response. The independent
verifier was corrected to consider only cancellations before that turn's ASR final;
both arms were then reverified with the same verifier hash. No eligibility, timing,
PCM or promotion assertion was loosened.

The Chinese control also retains two immediate post-cleanup 3 MiB / 1% failures;
subsequent stable checks were 3 MiB / 0% with zero compute process. The following
fresh response reached the known generic no-video `StopIteration`, so only the
NS-01-specific raw verifier is used for its 43/43 correctness result. Neither the
GPU assertion nor the generic postprocessor was weakened to manufacture a pass.

Full NS-04 still requires physical acoustic and human evaluation. No deployment,
resident-service change, training, optimizer work, generalized performance claim or
full two-GPU R4 rerun is claimed.
