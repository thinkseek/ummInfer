# NS-05 first-dynamic image establishment before native I2V — Task #90

## Decision

**NOT_PROMOTED; final default runtime candidate REMOVED.** The single director
candidate produced the required real NS-05 image→I2V chain, but the same frozen
Chinese long-form control regressed from 43 narration legs to 24 and changed the
accepted clean-text hash. That is a prompt/semantic non-regression failure, not a
successful promotion. Commit `79219f6400aa21a81be4df918846ebb9ca35886b` restores
the exact accepted baseline director. No second prompt variant was attempted.

The independently verified target pair is retained as negative promotion evidence,
not discarded. Its `PROMOTED_CORRECTNESS` field applies only to the isolated
NS-05 image→I2V target comparison; the final task verdict below is NOT_PROMOTED.

## Scope and frozen candidate

This task addresses the precise failure retained by
[Task #88](ns05-video-voice-cocreation.md): after a natural cyberpunk-animation
voice request, the model emitted video `slot-0` with `source=slot-0`. The engine
correctly rejected that self-reference, so no real image or I2V was generated.
The accepted baseline is `0645ec8ccf6dce668aedc4a214bf7da5e7567d32`.

The single candidate execution is
`2263bc64569abb61265f52e2fa4c4bb6c828d20e`. Its only production-source change is
in `src/pipeline/narrative_prompt.cpp`, scoped to auto narrative routing with
`continuity_mode="i2v_chain"` and at least two slots. When the first desired
visual is dynamic and no earlier image exists, the director now explicitly asks
for this ordered pair:

```text
[[image: <exact scene and style to animate>]]
[[video: <motion and camera description> | source=slot-0]]
```

It also explicitly forbids `source=slot-0` on video `slot-0`. No parser,
orchestration, public schema, model, sampling, native arithmetic, scheduler,
training, optimizer, or video-speed change was made. Invalid explicit sources
are never rewritten into unconditional T2V.

[Preregistration](../benchmarks/ns05-first-video/preregistration.json) froze the
single candidate before any new model observation. No prompt variant followed
the observed output. Test/harness commits are
`c6bf1c6d1e36f219ff69d95f51a5c6ad40430a8d` and
`12fbc4fcb72d44185cd4ed3281d57c37b25cf91b`.

## Frozen real Before/After protocol

Both measured arms used only explicitly idle physical GPU 7, fresh serial stacks,
the same native ASR/VLM/TTS/Wan binaries and FLUX stage, and the same toolchain
for their two gateway builds. They reused Task #88's:

- 120.000-second, 3,600-frame, 768×432 CC-BY-4.0 source video, 19,821,335 bytes,
  SHA-256 `ffef335edc161a2f6dead512e0fde96cf48fb997ae6b6491067e36c76eec58d0`;
- three real native-TTS recordings, resampled once to PCM16 mono 16 kHz and then
  actually transcribed by native ASR, with no transcript injection;
- identical request bodies, greedy seed `510075`, `temperature=0`, `top_p=1`,
  modality/render settings, cancellation at the autonomous visual slot, and
  immediate identical retry in the same session;
- zero client-authored directives, modality/frame switches, or direct-render
  fallback.

The retry request SHA-256 is
`621ab4b4ad0129fe0a04ff97249d301840e972bc3552b59608bbfcefb08b2938` in both
arms. Its exact ASR transcript is “Stop. Turn that driver scene into a cyberpunk
night animation and narrate it.” The initial phone and interrupted scene request
identities and transcripts are also preserved. All four public response
identities are distinct within each arm.

## Paired result

| Measure | Before | After |
| --- | --- | --- |
| Autonomous retry directives | video `slot-0`, explicit self-reference | image `slot-0`, then video `slot-1` with explicit source `slot-0` |
| Reference result | failed: `source slot slot-0 is not an earlier slot in this request` | completed, exact content-addressed binding |
| Real FLUX images | 0 | **1** |
| Native Wan I2V artifacts / submits | 0 / 0 | **1 / 1** |
| Clean retry text bytes | 84 | 358 |
| Public retry PCM chunks / bytes | 14 / 345,600 | 72 / 1,846,080 |
| Public retry PCM SHA-256 | `61ee204fd69d839dbfafb3935b98196c7f8acf0e3090d01d3e79bf54ea2e9162` | `78ac7036c6d80e466a849b30f36832ec6e9a7c7165a3a54477dc1d9c55bb3d8c` |
| Public terminal / final `in_flight` | complete / 0 | complete / 0 |
| Stale or cross-session events | 0 | 0 |

After's styled image is a decoded 512×512 PNG, 412,366 bytes, SHA-256
`0e2f53e1c26422fba627722b382b5899d7822d6a920b4e2a6bdefdc2100b8686`.
Video `slot-1` names source slot 0 and carries that exact anchor SHA-256. Its
native MessagePack submit is 413,409 bytes, SHA-256
`fd1e5b2cf8dec2c19cb9932973606b8e585286648d8e55926ee598350e2924ff`,
with `anchor_explicit=true` and `anchor_crop=center_cover`; independent decoding
proved the binary `inputs.image` equals the public source PNG byte-for-byte.

The output is H.264/yuv420p, 704×480, 29 frames at 16 fps, 285,195 bytes, SHA-256
`a7283fa962ba77c41cfc38acc83d9945c622ea1ec6156b049a870333832d4d78`.
The unchanged center-cover 64×40 RGB proxy gives first-frame MAE **3.356380**
and frame-0/frame-28 motion MAE **23.595182**. These are numeric continuity and
nonzero-motion checks, not perceptual-quality claims.

The successful After-r2 text, PNG, and MP4 are byte-identical to the earlier
After-r1 generation. After-r1 is retained separately because its cancellation
journal raced the response id; it is not the accepted journaled arm.

## Timing and NS-04 boundary

| Milliseconds, independently recomputed | Before | After |
| --- | ---: | ---: |
| Initial real-video question → first public text | 2,401.778 | 2,498.157 |
| Initial real-video question → first public voice | 2,633.735 | 2,729.582 |
| Retry start → first public text | 4,711.578 | 4,569.310 |
| Retry start → first public voice | 4,981.213 | 4,986.980 |
| Barge marker → interrupted scene terminal | 85.351 | 84.093 |
| Barge marker → initial co-creation terminal | 4,697.471 | 4,549.312 |
| Full barge → immediate-retry first public audio | 10,018.257 | 9,875.232 |

The initial real-video first-text and first-voice gates remain **below 3 seconds
in both arms**. The full replay/ASR/VLM/TTS barge-to-new-audio path remains far
above the strict 800 ms NS-04 target: **NOT_MET**, not hidden by the successful
visual correction. Physical microphone/speaker and perceptual listening remain
NOT_RUN.

## Fail-closed and compatibility controls

The new deterministic first-video control submits
`[[video: first dynamic | source=slot-0]]` and requires zero image/video native
submits, zero artifacts, one slot-attributed failure with the exact error above,
and one public terminal. Existing tests continue to cover:

- valid immediate and delayed image establishment followed by exact-byte I2V;
- malformed, empty, duplicate, missing, forward, self, wrong-modality, and
  cross-session references;
- failed or cancelled sources, transitive dependent settlement, stale late
  events, and valid recovery;
- existing explicit image-edit/I2V chains and ordinary text-only and explicit
  `source=slot-N` requests.

The scoped director test failed in exactly two assertions on the unmodified
baseline and passed with the candidate. The first-video self-reference and all
existing malformed/failed-source controls passed unchanged in that red run. The
raw red log SHA-256 is
`2400660a70ce08450613022fa05b42bfc22de188cf27f6735a47a861187ec565`;
the green unified log is
`a94f3d369d5055d037302b6ce7006d143cb95e808b7e6bbad2c3a9b543740057`.

A separate native host-test correction updates a stale Task-81 assertion:
reserved unclosed directives are removed, and a token straddling prose and a
directive must fail closed. Baseline and candidate both failed the old
assertion. No native production code changed.

## Exact execution identities

| Object | SHA-256 |
| --- | --- |
| Before gateway | `3c13bb1e3dd4bffcacdc1861e330d2b4b24c8ac80a9b4b415a223afa32398634` |
| After gateway | `a8ccc4d6e5e2650504ce26ddcf7d147969fe4207ccf38fbf703af5aa6342ea8b` |
| Native ASR | `e936045361d1f37ffe22717974dab9705afbb7c1e11023c44b003267afa42f86` |
| Native VLM | `686ff04d28c41eb7f34ffe3b80bffa73941f9776a9412df002a5b6a6bcef7500` |
| Native TTS | `92fd8e24ddfb3cf7a21748ade17db719c33b2d4633c4e1b580a94b7f5b500c91` |
| Native Wan | `21bfd59101e609345f6128cf9eaad918890376c5e16f44843a61c504e1ffa3f0` |
| FLUX stage | `ab88ae88f3a9207151d4774e4bb000fd385353724e22e7720cf94345a922ef74` |
| Candidate director | `dfdb8182cbeaff64739c49cd4ee60dba299a69c023471781fb81bb7253b099ea` |

Exact model-tree identities are Qwen3-ASR
`2b634d66b56817e11496d76402b9c282fb59235ff1a5df22a4e5e261f8bf1d06`
(13 files), Qwen3.8 `22e638404d0e413007064c47d5c98ad1696a26a0e980eb2fce83cbbfcca0a304`
(29), FLUX.2 `10c1b2d7096cd6c2798fa9d2b9aecfab5163117cbf3d66de6f32f9a2e8f17997`
(25), CustomVoice TTS `69c90c52d9d88390f4b4f540b09acf40f5ead7080650585ff20d2e144aa5e69d`
(14), and Wan2.2 `b118ac07b533cf7d68a4806ec70bdf39041515673e44535afeda5cfefe761616`
(30). Revisions remain `UNKNOWN`; exact local trees, not an upstream revision
claim, are the identity. Torch 2.6.0, Diffusers 0.43.0, Transformers 5.6.0,
pyzmq 27.3.0, and msgpack 1.2.4 were sealed in the execution record.

The independent paired verifier imports no live producer. It rebuilds raw SSE,
public media, native MessagePack source bindings, PCM/native/terminal-WAV joins,
ASR/request identity, decoded dimensions, continuity/motion, cancellation/retry,
cleanup, and all binary/model/source hashes. Two fresh processes produced the
same compact scorecard SHA-256
`4701d990ee7d0aa4f74f345bfae4a7a4a557a8329d6f463ec7da1884b2b2e171`.
The standalone After raw-verification receipt is
`485bb865f96bccd2b5446bea779d34ab3b6cafc48b5e28c526e407997c212774`.

## Retained failed attempts and corrections

- The first local-test attempts used the non-WebSocket system libcurl and were
  rejected as environment failures. The established WebSocket-enabled build
  produced the focused red/green result above.
- Before-r1/r2 did not produce an accepted model pair: the system Python
  resolved Torch CUDA 13 against the container's CUDA 12.2 driver; r1 also hit a
  transient `3 MiB / 1%` NVML cleanup sample. The accepted environment uses the
  established CUDA-12.4 FLUX virtualenv and bounded repeated idle checks.
- Before-r3 was a valid negative result but its journal recorded a null initial
  response id. After-r1 generated the correct real chain but had the same
  journaling race. No evidence JSON was patched. An observation-only wait now
  records the actual cancelled id before writing the journal; Before-r4 and
  After-r2 are the complete accepted pair.

Raw media, raw SSE, per-event payloads, MessagePack, PCM, logs, failed attempts,
and complete external manifests remain outside Git. The immutable paired record is
[ns05-first-video-pair.json](evidence/ns05-first-video-pair.json); the final verdict
and controls are in [ns05-first-video-sm90.json](evidence/ns05-first-video-sm90.json).

## Blocking NS-01 semantic regression

The candidate's frozen Chinese long-form request completed a real image and
source-bound I2V with 24/24 natural-EOS narration legs and final `in_flight=0`,
but it did not preserve Task #81's frozen long-form lifecycle. Its clean text
SHA-256 is `9ff5a2a4d303ef184727bacf004fe35677e644356dee6d3089b5b86eff173aad`,
not accepted `883802d889885bd032b18a3dc21c8f7e1ce77773f020ab0ee3e69a816ac8b572`,
and it has 24 rather than 43 native TTS legs. The exact Task-81-specific verifier
rejects it with `frozen Chinese clean text`. Valid media and 24/24 EOS do not
erase the changed long-form behavior, so this is the blocking regression. The
candidate raw report and independent media/PCM verification are sealed at
`5d6bed11ba6585310595f81845c2ea9bf7ced7d2b83ee5d983cc0fd251ad5bb3` and
`a4f6e6093dbcbc302846cad9ca8d7269f7560ffa4c3efa7dee95c54a1afef995`.

The restored baseline repeated the same frozen request with **43/43 natural EOS**,
the accepted clean-text SHA-256 and aggregate PCM SHA-256
`2ce2c53454349fd70edceab97e6afe6d93cf8ffc22e7bcdf497c303222d94f78`.
Two fresh Task-81-specific raw verifications were byte-identical at
`fe8f1c03a5aa614bf68a5152ee99d8201d8d940b44b0228c1be51b906fb9de4b`.
The generic postprocessor again reached its documented no-video `StopIteration`;
it is not counted as a model failure or used in place of that raw verifier. No
request text was shortened or rewritten to obtain either outcome.

## Deterministic, native, and protocol checks

The following checks passed without substituting them for real model/media gates:

- Debug and ReleaseFast `umm-test-local`, including the new director and
  first-video fail-closed controls, narrative/SSE replay, public PCM/network
  fixtures, cancellation, playback ACK history, queue bounds, and native/public
  arithmetic contracts;
- native Stage host CTest **11/11**, Qwen host/video-input CTest **3/3**, and
  FLUX contract tests **16/16 with zero skips** in the established remote
  Python environment;
- native video contract-stub completed/cancelled/failed/recovery behavior with
  exact terminals and final `in_flight=0` (explicitly not real Wan evidence);
- NS-03 frozen scoring contracts **12/12**, Task-73 first-audio contracts
  **6/6**, NS-04 replay **7/7**, public PCM **21/21**, realtime network **12/12**,
  SSE replay **11/11**, I2V geometry **13/13**, Showcase **28/28**, Web UI,
  cross-slot/NS-06 verifier self-checks, service lifecycle **13/13**, benchmark
  replay **70 cases**, NS scorecard **4/4**, and NS-08 zero-control **7/7**;
- feature-map validation at the accepted starting baseline and a repository-wide
  relative Markdown link check covering 90 files and 329 links.

Final rebased feature-map, link, live non-regression, and external-manifest
results are recorded below after their execution.

## Final non-regression outcome and limits

- Development NS-03 completed 20 frozen cases per arm with identical outcomes:
  `15/20` routing, one redundant visual, zero empty prose, zero missing voice,
  zero public leg errors, and `2/5` follow-up context. Independent raw receipt
  hashes are `450534ce…90f3` Before and `a5f94368…b429` After.
- The held-out Before arm was stopped during recovery after eight raw streams.
  It is **PARTIAL_NOT_SCORED**, not a passing 20-case arm. Its recorded harness
  PID received SIGINT and its own `finally` path stopped the stages; GPU 7
  returned to `3 MiB / 0%` with zero compute processes. Remaining held-out arms
  are NOT_RUN after the rejection and restart.
- Fresh independent processes reverified the accepted NS-06 raw pair instead of
  duplicating generation: 3/3 keyframes and I2V, two exact edit links, three
  exact I2V links, and equal prose/ASR/PCM. Receipt SHA-256 is
  `c2685c4b0f727fa47fcfee297292bf16d7c3de9d4cb954f80afdae559182549e`.
- Fresh public PCM verification passed 25 HTTP responses (24 complete plus one
  cancel), 29/29 realtime responses and final ACKs, 493 exact legs, 4,176 chunks,
  115,034,880 bytes, and the exact 4,096 pending-audio release. Receipt SHA-256
  is `389104a350b56757ef0dec2d66c172d2474bd8021d3a506f3e59b23120fdc4e8`.
  Fresh realtime-network verification passed six sessions, three rounds, 41
  responses, 30 exact next-turn prompts, 32 terminal callbacks, and final
  `in_flight=0` (log `79f70e82…1cf0`).
- Accepted public narrative raw reconstruction passed 69 events, 32 artifacts,
  exact FLUX/I2V source bytes and five PCM legs (receipt
  `1847b98e08581f198010ce568950af7b4e6b6e7a4e8a6606731d7a762e268dd9`).
  These accepted raw controls were not regenerated after the candidate had
  already been rejected.
- Final restored Debug and ReleaseFast `umm-test-local` logs are
  `c3840c15e03a43975724449043bbc16c158c45ae1cacf6c22515fedcc57e4802` and
  `68ad59a97a43cce2d40dcd6010487e0961cf1fdbede538ac4acc12e097069eb8`.
  Native Stage host 11/11, Qwen 3/3, FLUX 16/16 zero-skip, and native video
  lifecycle contracts pass.

The machine-readable final verdict is
[ns05-first-video-sm90.json](evidence/ns05-first-video-sm90.json). Human
perceptual review, physical acoustics, full production, full two-GPU R4, and
remaining held-out NS-03 arms are NOT_RUN. The default first-dynamic defect
therefore remains open: this task proves one candidate's real completion gain
and rejects it for a measured semantic regression rather than promoting it.

## External evidence seal

The complete external self-excluded manifest covers **3,562 regular files and
3,302,872,470 bytes**, excluding only `SHA256SUMS` itself and non-regular runtime
sockets. Its SHA-256 is
`1f2d3442008c7963c3d1cbd252568a1e34d9cc90b766ab14c372410becf8b772`.
Two fresh manifest-check processes produced byte-identical PASS receipts at
`910a4004c0c85a78582fd175e2ba62b3da327b179e4127d8da47b73fd1afd318`.
No sealed evidence was changed after those checks.

All task-owned processes are stopped. Final GPU 7 state is **3 MiB / 0%, zero
compute processes** and public health is `in_flight=0` in the applicable live
controls. GPUs 0 and 1 were not used, GPUs 2–6 and the resident service were not
contacted or restarted, and no push or deployment occurred.
