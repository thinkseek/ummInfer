# Realtime ASR-final to first-audio scheduling investigation (Task #55)

## Scope and evidence classes

This document separates three kinds of evidence:

1. accepted Task #41 R3 retained bytes, independently reanalyzed by Task #53;
2. Task #55 diagnostic runs, including failed or resource-contaminated windows;
3. the final paired, quiet-window baseline/candidate gate, reported separately below.

All measurements are digital loopback server/native-model measurements. Physical
microphone/speaker playback, device jitter, WAN behavior, acoustic quality and
perceptual listening are **NOT_COVERED**. A scheduling improvement is not a claim
that a user hears sub-second audio on a physical device.

The work started at integrated Wave 71 commit
`22d34659a51f1c91216c2748387521b4f0f5ac6d`. The measurement harness commits are
`5d5e7c9382fe5ac29b80c31ef8c381073a421de3`,
`9632321093b4590dff078e43b148a9916f70086a`, and
`f564ccee5b7df8f7b2441b223d477741777117ff`. The candidate scheduler commit is
`cab7894c5c8dd98d9b171645e780b4be7408d776`; the bounded-lossless protocol commit is
`6dc6dfce66a975c1927e95fb3f1d93cfeb2cac49`. Final integration merge
`bab21ee7fbe021220da6cc80260750f12ca7fbc8` has second parent official Wave 80
`bb1ecaa6f38526434066df2eaf945076a72a8ac8` and preserves the earlier measurement
history without altering the independent ASR allocator implementation.

## Retained R3 decomposition, before any new GPU launch

The sealed Task #41 R3 evidence was copied into a fresh scratch directory. All
44 `SHA256SUMS` entries passed before analysis. The accepted Task #53 auditor
recomputed the committed report in a fresh process with byte-identical SHA-256
`7cda1b7f08db2047430630cf59b64ba55e19dc2ccab7fcce0eca54bab058fddc`.
The earlier live driver summary was not used as a timing oracle.

A separate offline join decoded the raw WS frames again, used the first public
`audio.delta.text_end_cursor` to find the exact text delta that made that segment
available, and joined native ASR/Qwen/TTS log durations. This does not recreate
the Task #53 audit: it adds stage attribution to its retained raw evidence.
Durations below are milliseconds, computed from integer nanoseconds before
conversion. Native log decimal-ms values are kept at their logged resolution.
The signed ASR-final/prompt offset preserves the two original recorder threads;
it is not clamped to zero.

| Stage metric | n | Median | p95 | IQR | MAD | Min / max | Tukey outliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native ASR final compute | 30 | 44.000 | 45.100 | 1.000 | 0.000 | 24.000 / 47.000 | 5 |
| ASR speech-stop → final | 30 | 52.040 | 212.947 | 41.324 | 6.850 | 33.822 / 271.063 | 3 |
| ASR final → observed VLM submit (signed) | 30 | -0.059 | 0.066 | 0.076 | 0.040 | -0.427 / 0.110 | 2 |
| VLM observed submit → first text | 30 | 820.389 | 6724.914 | 2733.221 | 519.049 | 301.267 / 9312.296 | 2 |
| VLM first → last text | 30 | 1407.346 | 3015.030 | 537.657 | 210.560 | 0.000 / 3508.167 | 4 |
| First text → first-audio segment available | 25 | 70.315 | 80.669 | 30.241 | 10.040 | 20.049 / 80.847 | 0 |
| Segment available → first WS audio | 25 | 12615.177 | 23357.910 | 17844.186 | 7153.237 | 180.895 / 24170.566 | 0 |
| First TTS compute → first PCM (native log) | 21 | 72.910 | 185.141 | 111.407 | 1.651 | 71.260 / 185.520 | 0 |
| Segment-to-audio residual minus native first PCM | 21 | 12492.271 | 20629.794 | 16547.682 | 7094.166 | -3.212 / 23512.522 | 0 |

The small negative residual minima are recorder/log-resolution effects, not
negative latency. The retained run has no exact TTS-submit timestamp, so this
last row is only a bound on the combined admission/setup/transport residual.
The 0.181–0.191 s segment-to-audio intervals in the six single-active-response
echo/double-talk cases contrast with multi-second concurrent cases. Sentence
segmentation and native first-PCM setup therefore do not explain the retained
17.291 s median TTFA.

## R1 diagnostic: exact native-boundary observation

The new driver places byte-preserving ZMQ forwarding observers at ASR submit,
VLM submit, TTS submit, and the shared native completion edge. Every forwarded
frame retains its original bytes, SHA-256, receive monotonic timestamp and
forward timestamp. Raw public WebSocket frames and PCM are independently decoded
by `python/umminfer_realtime_first_audio.py`; the verifier does not import the
live driver.

R1 used the exact baseline binary set:

| Binary | SHA-256 |
| --- | --- |
| `umm-infer` | `318eecd2ae77e31846d22aa6df767199d4f9bb5a295407ee57c78bd02e452689` |
| `umm-native-stage-asr` | `822a9c4a44bff6bb54f62784321a50fd12265dbb77a5dd7a626ea0a78efd0ffc` |
| `umm-native-stage-qwen38` | `d7b74d788a098763e0c79b18306f1bda276415e48bb4ad8a995ebc04bf15e08f` |
| `umm-native-stage-tts` | `6270aa60384d688f70d381dd1a215c80f661ac79692ccf81baa60a4464458396` |

In the 18-sample `concurrent-cold-short` diagnostic arm, raw timestamps gave:

| Metric | Median (ms) | p95 (ms) | IQR (ms) | MAD (ms) | Min / max (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ASR final → first WS audio | 17815.649 | 29795.968 | 13989.829 | 6835.955 | 371.476 / 29810.091 |
| VLM submit → first native text | 4378.449 | 8617.429 | 5112.153 | 2555.832 | 149.723 / 8621.201 |
| First native text → first segment | 35.246 | 39.586 | 8.459 | 4.309 | 17.839 / 39.587 |
| First segment → TTS submit | 34.046 | 39.704 | 4.794 | 2.612 | 17.943 / 39.795 |
| **TTS submit → first native PCM** | **13351.400** | **21099.581** | **8882.011** | **4269.865** | **185.357 / 21101.083** |
| Native TTS compute → first PCM | 72.158 | 182.307 | 1.026 | 0.448 | 71.610 / 182.503 |
| First native PCM → first WS audio | 4.610 | 9.973 | 5.689 | 2.824 | 0.560 / 10.203 |

Every slow response remains in these statistics. Two early-position requests
completed TTFA in about 0.38 s; later requests waited up to 21.1 s after their
first TTS submit even though their own native first-PCM work was about 72 ms.
This directly proves a serial-TTS admission/ordering defect: continuation legs
already queued by faster responses run before another response's first leg.
Audio transport and sentence availability are not the multi-second source.

R1 later stopped in `serial-cold-long`: the experimental `zh_16.wav` fixture
produced an empty repeated-turn ASR final, so no VLM/TTS response was expected.
The driver failed closed instead of inserting a zero or dropping the turn. Its
first four arms retained 3 + 4 + 18 + 24 driver-recorded turns; the failed arm
has no accepted measured sample. Both assigned GPUs were independently observed
back at 3 MiB, 0% utilization, with no compute process. R1's initial top-level
cleanup flags incorrectly stayed false on this error path even though per-arm
postflight succeeded; commit `f564cce` fixes that reporting aggregation. R1 is
diagnostic, not a completed controlled gate.

## Candidate scheduling rule

The candidate adds no public field and no second transport protocol. An empty
`Stage::Config.request_priority` preserves legacy behavior. Only the native TTS
executable installs a predicate for the existing internal child ids:

- streamed first leg `:tts:0` and one-shot `:tts` form the first-audio class;
- later `:tts:N` continuations and ordinary requests remain FIFO;
- each class is independently FIFO;
- when both classes are queued, the serial queue alternates them, ensuring that
  existing continuations continue to drain instead of starving;
- the hook is rejected for batched/concurrent scheduling and after start;
- abort tombstones, request ownership, public PCM/cursors, and terminal behavior
  are unchanged.

`native/stage_cli/first_audio_scheduler_test.cpp`
covers the original inversion, fair continuation progress, stable class order,
abort-before-submit, and a no-hook strict-FIFO control. The full local suite and
realtime stress target are also required. An additional 20-iteration, 12-session
run passed 3,360 scheduled actions with queue bound 4,096 and trace
`0xa8d7d66c5bedf741`; the dedicated ASan/UBSan build produced the same PASS and
trace. A first direct build with `/usr` prefixes selected
incompatible system curl/httplib headers and did not run a test; the recorded
pass uses the documented compatible dependency prefixes.

## Prospective controlled contract

`benchmarks/realtime-first-audio/preregistration.json` is the frozen seven-arm
contract. All arms use the same native models, public WS topology, VAD, 96-token
VLM limit, 192-frame TTS limit, `vivian` voice, streaming PCM16 24 kHz mono,
session history and playback ACK protocol. Only the named arm factor changes.
SplitMix64 seed `5630241` fixes per-round session order and jitter. Three measured
repetitions produce 81 measured responses; warmups and deliberate regression
turns are retained but excluded from primary pairs.

| Arm | Sessions × measured rounds | Prefix | Fixture / consumer behavior |
| --- | ---: | --- | --- |
| `serial-cold-short` | 1 × 3 | disabled; require reuse 0 | retained short microphone recording |
| `serial-warm-short` | 1 × 3 | enabled; one excluded warmup | same recording, same session |
| `concurrent-cold-short` | 6 × 3 | disabled; require reuse 0 | primary arm |
| `concurrent-warm-short` | 6 × 3 | enabled; one excluded warmup/session | warm history and prefix reuse |
| `serial-cold-long` | 1 × 3 | disabled | alternate established English ASR recording |
| `concurrent-cold-long` | 6 × 3 | disabled | alternate recording under concurrency |
| `concurrent-cold-short-backpressure` | 6 × 3 | disabled | 2 s read pause after recording first raw audio; 250 ms final ACK delay |

The alternate long arm uses `en_16.wav` SHA-256
`9fe58914522a29ac7afde36af931f83b028f6540f23b29bb06d48ae1f98a1d4a`.
Its actual first-segment byte length is reported rather than assumed. The failed
R1 Mandarin candidate is not silently substituted within any accepted run.

The primary A/B pairs responses by arm, logical session and measured round.
Paired ASR transcript, complete text, prompt history, first TTS text/cursors and
concatenated PCM SHA-256 must all match. The primary median must improve by at
least 20%, primary p95 may not regress more than 5%, and serial median may not
regress more than 5%. Population CV is computed as standard deviation/mean. Both
serial arms require CV <= 0.10; concurrent CV is reported but not gated because
seeded admission positions intentionally create heterogeneous queueing. Every
outlier remains visible. The separate delayed-barge-in, echo-only protected/
disabled A/B, and simultaneous double-talk block checks exact acknowledged-prefix
history, cancellation and isolation.

R2 used the corrected harness and completed the six non-backpressure arms, but
its window overlapped a remote CPU-only candidate compile. Before the final arm,
the long `concurrent-cold-short-backpressure` name made a Unix IPC path exceed
107 bytes and the stack failed before launch. The harness now uses short unique
owned `/tmp` IPC/SHM directories. No R2 backpressure or regression result is
claimed. R2 is diagnostic and cannot support the final timing claim. The final
claim uses only the later quiet-window R3 baseline and candidate described below.

## Quiet R3 baseline, independently verified

The R3 baseline ran all seven arms and the separate regression block without a
concurrent remote build or analysis job. Its unchanged raw bytes were subsequently
analyzed by the corrected independent verifier. The final report is
`fff44bed8dbd0144aeb99f355eedf8b0caf49d3c1bc1aea6853c3bf6836225b8`,
with 81 measured responses, 96 total server responses, 96 history turns, 380
accepted ACKs, four deliberate cancellation callbacks, and 96 completion
callbacks. Every measured response completed, received full-text `audio.done`,
and received a final ACK. Protected echo produced zero raw cut-ins; delayed
barge-in and both double-talk modes passed. `/health` ended at `in_flight=0`, all
owned processes exited, and both assigned GPUs returned to idle.

The first verifier attempts failed closed on the reliable abort PUSH, a legal
late native completion after abort, the whole-file JSONL cap, and the VAD/TX
capture boundary. The warm-tail failure at `concurrent-warm-short:req-8` is
retained as a separate failed analysis: the committed PCM prefix must be joined
using `speech_stopped.audio_ms * 32`, while the post-commit tail remains available
to the next warm turn. Verifier commits `238e410`, `c5f99b2`, `911b771`,
`78ae321`, and `cd4a909` address those evidence-decoding issues without changing
the runtime, raw frames, or measured clocks. Only after those corrections did the
same raw baseline pass. No failed analysis is counted as a model or timing pass.

| Baseline arm / stage | n | Median (ms) | p95 (ms) | IQR (ms) | MAD (ms) | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Serial cold short: ASR final → first audio | 3 | 381.717 | 390.721 | 5.115 | 0.226 | 0.0124 |
| Serial cold short: TTS submit → first native PCM | 3 | 187.580 | 188.747 | 1.276 | 1.255 | 0.0056 |
| Concurrent cold short: ASR final → first audio | 18 | 18136.436 | 29950.429 | 14183.127 | 7095.424 | 0.5637 |
| Concurrent cold short: VLM submit → first native text | 18 | 4405.148 | 8651.846 | 5134.719 | 2569.533 | 0.6611 |
| Concurrent cold short: first text → first segment | 18 | 35.493 | 39.780 | 7.194 | 4.127 | 0.2271 |
| Concurrent cold short: TTS submit → first native PCM | 18 | 13648.843 | 21218.944 | 9048.301 | 4522.825 | 0.5373 |
| Concurrent cold short: native compute → first PCM | 18 | 72.239 | 183.584 | 1.096 | 0.434 | 0.4565 |
| Concurrent cold short: native PCM → WS audio | 18 | 7.367 | 9.441 | 4.765 | 1.710 | 0.4274 |
| Concurrent warm short: ASR final → first audio | 18 | 23748.078 | 41345.811 | 15994.587 | 8733.468 | 0.5972 |
| Backpressure: ASR final → first audio | 18 | 17648.588 | 28899.830 | 13228.046 | 6618.288 | 0.5568 |

Warm measured prefix reuse reached 244 tokens. The backpressure arm's two-second
consumer pause starts only after the first raw audio frame has been recorded;
it is not included in first-audio time. Its later raw PCM/ACK and terminal
contracts still pass.

The replacement `en_16.wav` recording transcribes as `Hello.` and deterministically
produces the 6-byte first segment `Hello!`. It is therefore a useful alternate
short-opening arm but **does not establish genuinely long-opening live coverage**.
That limitation is retained explicitly rather than relabeling the arm. The
existing sentence splitter's host tests cover long punctuation-free input; a new
real long-opening prompt remains outside the measured claim.

The first candidate window completed its first six arms, then refused to start
`concurrent-cold-short-backpressure` because GPU 0 reported 6% utilization with
3 MiB and no compute process. No last-arm launch occurred. That partial candidate
is diagnostic only; its raw evidence is retained, and a fresh full candidate with
identical source/binaries/plan is required after both GPUs return to explicit
idle. No partial-candidate latency improvement is claimed.

### R3 paired quality failure and bounded-lossless correction

The fresh full candidate itself passed both independent verifier processes and
sealed 235 files. Its report SHA-256 is
`c23624a138637aaa5829cf6b569579770685822b1a5ff015880e0e386633a729`;
the candidate manifest SHA-256 is
`433f53e6094ba49a83e2051ae4e23ba10be940215b3904a30d8532b7824c5295`.
The baseline report/manifest SHA-256 values are respectively
`fff44bed8dbd0144aeb99f355eedf8b0caf49d3c1bc1aea6853c3bf6836225b8` and
`bfd31661f82abdff597e1a102ccb1535e528c4b66d25308a429e8b57353b9259`.

The fresh paired comparator nevertheless returned **FAIL**. Text and prompt
identity matched, but public PCM did not. In `concurrent-cold-short`, five
candidate pairs lost public chunks while the baseline pairs were complete; in the
backpressure arm, six baseline pairs were short while the candidate was complete.
Other warm/backpressure pairs also differed. Thus neither ordering can claim a
quality-safe improvement from this evidence. Both runs ended with GPUs 0/1 at
3 MiB, 0%, and no compute process.

The native observer bytes exposed the shared protocol defect: at 4,096 queued
unified events, `RealtimeApi::forward_event` dropped incremental VLM/PCM events,
while terminal text/ACK progression continued. The correction blocks incremental
producers until `drain_unified()` swaps a bounded batch, keeps terminal events
outside that wait, and wakes blocked producers on `stop()`. At most the 4,096
queued events plus one drained batch are held. This is bounded backpressure, not
an enlarged unbounded queue.

The analyzer now also requires every measured response's PCM to match native TTS
per leg, keyed by response id and actual `(text_cursor, text_end_cursor)`. The
decoded UTF-8 slice at those cursors must equal that leg's submitted text, and the
cursor chain must cover the full response. A deterministic missing-chunk fixture
fails this check. R3 remains a failed A/B diagnostic; only the subsequent common-
server R4 comparison can support an improvement claim.

An initial R4 baseline launch used a misspelled TTS model directory and stopped
when that task-owned TTS process exited during loading. No response or timing
sample was produced, the failed directory is retained, and both assigned GPUs
returned to 3 MiB / 0% with no compute process. The measured R4 baseline uses a
fresh output tree with the pinned `Qwen3-TTS-12Hz-1.7B-CustomVoice` directory.

### Native synthesis repeatability controls

Six fresh-process native CLI runs of the identical greedy `It sounds` request
produced the same WAV SHA-256
`8a78dd86aa96e4d906a0832ba4fa34855768344dc6a78d044481f50bb5ea1037`.
A reused candidate TTS stage then produced identical PCM for six alternating
short-text requests and for six repetitions of the exact continuation text
`like you're in the middle of a thought or a`; the latter all had 737,280-byte
PCM SHA-256 `3eaa2791bba77428558a314086e2264e3279123b03d89a4998f24a3cce11773d`.
Thus both fresh-process and simple reused-stage controls are repeatable.

Per-leg R4 comparison found 17 differing response pairs. Every differing leg has
identical submitted text, and all are continuation legs rather than leg 0. Each
side independently satisfies native/public PCM identity after the lossless fix.
This does not prove order independence under the full queueing workload, so a
same-candidate A/A repeat is retained as a separate diagnostic. No accepted gain
or broad no-quality-regression claim is made from R4 while that control is
pending.

## Integrated R4 measured result and exact-quality limitation

R4 used final integration merge
`bab21ee7fbe021220da6cc80260750f12ca7fbc8` and candidate verifier source
`e123fa69492dc566f285e58fdc6d800985e21706`. Both sides include official Wave 80
`bb1ecaa6f38526434066df2eaf945076a72a8ac8`, the lossless-backpressure server, and
the newer ASR/TTS allocator implementations with default request behavior. The
common server/ASR/Qwen SHA-256 values are respectively
`91863dbda8de6480d232f863c542b6854818597fa06ea68219b7366b3e0e432d`,
`21c0dc097b0c958523496ae8e9c9f0c865eb436b8f15e7f121e24688baa568a5`, and
`a958db362ab0df68aeabe6ee258c0d4f0840a13a7657b03c59b0bda074aef474`.
Only TTS differs: FIFO `26e123d3562b5ade9217427015fb1740e3daaaea1905f74daae92bbe194b91c8`
versus fair-priority `ca36c5c68a15904ffaf425404351d3fa5b35936e014d65224f2946c19570bbc7`.
`BASELINE_POLICY.diff` is exactly the one-line predicate disable and has SHA-256
`70ce8c986cdb93c4c030b541cb0e908605cfad17543d7f707101df11fe7f99cc`.

Each of the baseline, candidate, and same-candidate repeat passed two independent
fresh-process analyses with 81 measured responses, 96 total server responses,
235 sealed files, exact native/public per-leg PCM, `in_flight=0`, and full process/
GPU cleanup. Their report SHA-256 values are:

- baseline: `9fcabf44eeb29c38cab3187c15f03c796755d96389735fad7c2e7bb152d3f511`;
- candidate: `cb8920590983126dd4d9fa6d6356680df2b5344de61b108b6160608505076a2d`;
- same-candidate repeat: `aa06c989979902f9df082cb741551fd423c74b606c1532f29b9b14ffcfb452a2`.

| Primary concurrent-cold-short metric | FIFO baseline | Fair-priority candidate | Same-candidate repeat |
| --- | ---: | ---: | ---: |
| Median TTFA (ms) | 18180.005 | 9696.090 | 9788.176 |
| p95 TTFA (ms) | 30086.814 | 11930.030 | 11936.008 |
| IQR (ms) | 14287.477 | 3766.227 | 3853.119 |
| MAD (ms) | 7145.024 | 1830.991 | 1858.527 |
| CV | 0.5647 | 0.4642 | 0.4611 |

The measured primary median reduction is **46.666%** and p95 reduction is
**60.348%**. The repeat independently lands at 9.788 s rather than hiding the
first 9.696 s window. TTS submit → first native PCM moved from 13,704.609 ms to
4,589.929 ms. Serial median changed only 381.59784 → 381.63647 ms (**+0.0101%**);
serial candidate CV is 0.01252. Warm concurrent median moved 23,694.939 →
10,108.531 ms, and the read-paused backpressure arm moved 18,191.092 → 9,797.515 ms.
These are measured diagnostic gains on the stated digital-loopback topology, not
physical-playback or WAN claims.

However, the preregistered exact-full-PCM comparison remains **FAIL**. There are
17 A/B PCM-mismatched pairs and 18 mismatches in the same-candidate A/A repeat.
All full response texts match, all differing legs are continuation legs with
identical submitted text, and leg 0 PCM does not differ in either comparison.
The A/A variability means a candidate-attributable regression is not established,
but exact full-waveform quality equivalence is also **NOT_ESTABLISHED**. Perceptual
listening remains **NOT_COVERED**. Therefore the latency numbers above are not
promoted to a quality-safe production acceptance claim. The original failure is
preserved rather than rewritten as a PASS.

Exact full reports and the machine-readable diagnostic verdict are:

- [R4 baseline](evidence/realtime-first-audio-r4-baseline.json)
- [R4 candidate](evidence/realtime-first-audio-r4-candidate.json)
- [R4 same-candidate A/A](evidence/realtime-first-audio-r4-candidate-aa2.json)
- [R4 diagnostic summary](evidence/realtime-first-audio-r4-diagnostic.json), SHA-256
  `885c03e01bf04872c9a063415ab1844405cd38d4eccdacc3853792cf37369902`

The diagnostic summary also pins raw manifest hashes for the failed R1/R2,
partial first candidate, invalid model-path launch, and all three native TTS
controls. Every diagnostic checksum passed. Final GPUs 0/1 were both at 3 MiB,
0% utilization, with no compute process. A genuinely long-opening real prompt is
still **NOT_COVERED** because the alternate recording deterministically produced
`Hello!`; this remains an explicit follow-up boundary.

## Wave 82 integration closeout

After the R4 evidence was sealed, follow-up integration merge
`aa8cd55d0a462f144b4fdac17ea35be459c2d32a` retained that complete evidence and
added official main `c793555695e4132aa61cca00fdc416e97c16457f`, which includes
the requested Wave 81 `ebbf254` client frame-budget work and Wave 82 Journey 3
GUI export alignment. No ASR or client implementation was modified by the
conflict resolution. A duplicate nonstream-artifact row introduced by the merge
was removed, and the Task #55 row remains singular.

The final combined tree passed `umm-test-local`, Web UI contracts, 28 Showcase
tests, 4 first-audio evidence fixtures, and 10/10 native host tests. Feature-map
validation at merge `aa8cd55` passes 214 rows and 1,056 references; the
hand-maintained AOCI structural validator passes 525 entries and 1,680 relation
targets. The Swift `String.init` issue remained owned and validated by the
mainline integrator; no local Swift implementation change was made here. R4 raw
evidence, diagnostic hashes, and the exact-full-PCM limitation are unchanged by
this integration-only closeout.

## Final disposition after supervisory review: blocked diagnostic

The frozen `ab_acceptance.no_fix_path` requires no latency runtime change when a
guardrail fails. R4 preserves 17 A/B and 18 same-candidate A/A full-PCM mismatches,
so the measured diagnostic gains cannot justify enabling fair-priority by default.
The final tree therefore reverts the scheduler/pipeline/native-TTS priority change
and removes its now-inapplicable host test. Historical candidate commit
`cab7894c5c8dd98d9b171645e780b4be7408d776` and every R4 report/raw failure remain
available, but no current runtime entry point uses that candidate.

The independently proven bounded-lossless `realtime_api.cpp` correction is retained.
The final runtime diff against official `c793555695e4132aa61cca00fdc416e97c16457f`
contains no fair-priority or other latency-scheduling change. No threshold, prompt,
raw sample, report hash, or outlier was altered to reach this disposition. The Task
#55 latency-optimization outcome is **blocked/diagnostic**, not a done quality-safe
improvement. Native continuation order-independence remains the follow-up blocker.

## Task #59 native continuation determinism closeout

Task #59 isolated that follow-up blocker without changing the frozen reports,
analyzers, scheduling, scores, thresholds, or fair-priority reversion. The repro
was derived against official `49d34063b7ecb61d0571bf962b11f1a71c0d039f`, and
final integration uses Wave 84 `ac25f5cc31b295c5a1df8399731dd1feee565190`.
A direct `TtsEngine`
replay used the exact R4 continuation text and native chunk schedule. Two
pristine fresh-process runs of the full 108-request cold-short order disagreed
on 2/108 requests; an eight-request prefix disagreed on 1/8 in both A/A
comparisons, while the sampled two-request control stayed equal. Generated
codec frames differed as well as PCM, so observer alignment is not the cause.
For `req-4:tts:1`, both pre-fix PCM records had 737,280 bytes but different
hashes, first differing sample 209,281 (1049 versus 1050), and their first
generated-code difference was frame 109, codebook 0. `req-9:tts:1` first
differed at sample 180,481 and produced 156 versus 192 frames.

The root defect was in `repetition_penalty_kernel`: one thread per emitted
history entry performed an unguarded read/modify/write on `logits[id]`.
Duplicate group-0 ids therefore raced and could apply the penalty repeatedly,
changing argmax and later EOS. Native fix commit
`b12d9026ded138a9fd3d6e623a657642157f7ac2` now builds a shared-memory id
bitset with idempotent atomic bit sets, then lets one thread per logit apply
the penalty at most once. This preserves Hugging Face gather/scatter semantics
and avoids an O(logits × history) decode scan. No public request or PCM field,
allocator ownership, cancellation path, or scheduler rule changed.

The focused real-CUDA fixture failed before the fix (mixed duplicates: 3
semantic and 5 repeat failures; dense duplicates: 12 and 7) and passed after
with zero failures. Two final fresh-process 108-request A/A runs had zero PCM
or frame mismatches across 45,676,800 compared bytes. Both prior `req-4` and
`req-9` continuation records became 737,280-byte PCM SHA-256
`3eaa2791bba77428558a314086e2264e3279123b03d89a4998f24a3cce11773d`.
Two actual native-stage runs each passed ten successful requests, exact
stream/terminal PCM, nonstream/stream equality for the tested continuation,
cancel/retry equality, and seeded fragment/rejoin identity. The independent
fresh-process verifier checked all these raw records and 1,479/1,479 checksums;
its machine-readable result is [Task #59 SM90 evidence](evidence/tts-continuation-determinism-sm90.json),
SHA-256 `a34832621ce0a9b151257394284fb9476b04d451b118fa71ca6d0023556a0b47`.

This establishes minimal native A/A byte identity only. A later frozen R4
exact-quality replay is justified before reconsidering any candidate, but the
full two-GPU R4 replay is **NOT_RUN** here, fair-priority remains reverted, and
the earlier diagnostic latency improvement is **not unlocked**. Public HTTP
speech is **NOT_RUN** because its required unified VLM entry is outside this
native-TTS task; the failed launch-only attempt produced no speech evidence.
Perceptual listening and physical playback remain **NOT_COVERED**.

## Task #63 public HTTP/native PCM follow-up

The remaining public-speech gate is implemented by
`tests/public_speech_pcm_live.py` and independently checked by
`tests/public_speech_pcm_verify.py`. It uses one isolated GPU 0 unified
VLM+TTS stack and retains raw public HTTP/SSE/JSON, realtime WebSocket, and
native MessagePack records. Fresh-process verification binds every public PCM
chunk and text cursor to exact native-stage bytes, checks repeats,
cancel/retry, seeded fragmentation, delayed reads, final playback ACKs, and
`in_flight=0` cleanup. The exact direct continuation control is not conflated
with the VLM's independently retained `you'r` output.

The delayed-reader release uses only native TTS audio minus public
`audio.delta`, with a hard 4,096-event threshold and a fail-closed timeout.
Text and control events cannot force release. This is a bounded-lossless and
byte-parity gate, not a rerun of the seven-arm/two-GPU R4 benchmark or a
fair-priority evaluation. Exact measurements, binaries, model/config/source
hashes, evidence checksums, failed attempts, and cleanup are recorded in
[public-speech-pcm-gate](public-speech-pcm-gate.md). Physical playback and
perceptual quality remain **NOT_RUN**.

## Task #70 realtime transport-validity lifecycle

The 12 delayed-reader resets were matched to libwebsockets' default 300/310 s
connection-validity hangup, not to native PCM loss or the unchanged 600 s
response deadline. Client PING frames do not confirm validity in that library;
PONG or an explicit confirmation is required. Execution commit `2b49b5b`
therefore holds transport idle expiry only while ASR/response ownership or
queued generation-complete output remains, and restores ordinary idle expiry
after release. It never refreshes the application response deadline.

A real-loopback PING=1 s / hangup=2 s regression failed at 2.001 s before the
fix, then passed with exact PCM and one final ACK; the independent 10 s stall
control still timed out and idle-only cleanup remained effective. The public
verifier now rejects anything other than 29 paused readers and exactly 4,096
pending audio events. See [public-speech-pcm-gate](public-speech-pcm-gate.md)
for the preregistration, red/green hashes, and subsequent real GPU evidence.
The Wave 94 R2 gate passed all 29 responses with one terminal/final ACK each,
493 exact native/public PCM legs, and the exact 4,125−29=4,096 delayed-reader
release. All 25 HTTP controls passed, two fresh reports matched, 1,266 evidence
files passed two manifest checks, and GPU 0 returned to 3 MiB. Full R4 and
physical playback remain **NOT_RUN**.
