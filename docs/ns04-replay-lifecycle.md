# NS-04 replay-inclusive lifecycle diagnostic (Task #82)

## Result

**NOT_PROMOTED; zero candidate runtime change.** The complete frozen
solar-system → Saturn-rings → switch-to-English journey is correct, but its
barge-to-new-audio maximum is about **2.879 s**, above the strict **800 ms** gate.
The accepted Task #81 runtime is retained unchanged. No partial-transcript reply,
shorter audio, faster replay, moved timing anchor, disabled history, extended
response deadline, new client control, training, optimizer, or video-speed
candidate was used.

The preregistered After arm is deliberately a **same-source diagnostic A/A**
because profiling found no semantic-preserving C++ serving/realtime/scheduler
change that could overcome the measured input/replay interval. The scorecard
therefore records `same_source_diagnostic_A/A` and the full-path threshold failure;
it does not claim a latency improvement.

## Frozen contract and inputs

[Preregistration v2](../benchmarks/ns04-replay-lifecycle/preregistration.json)
has SHA-256
`caa4d56171ece814b298fd9764ec3dadd5aa1f860cdf6d947582ac811b298f80`.
The first contract was committed as `4b340e1ac34c704e4e83dadd36c83eda00a9aa77`
before new measurements. Revision 2 was committed as
`c83abc4` before any paired run and links the v1 hash.

All three frozen WAVs are PCM16 mono 16 kHz and are actually transcribed by native
Qwen3-ASR; no transcript is injected. They were prerecorded once with the
unmodified native TTS executable, not captured from a physical microphone.
Each arm uses the same complete files, 3,200-byte chunks, chunk-duration pacing
(100 ms for full chunks), six 100 ms end-silence chunks, 400 ms VAD silence,
768 VLM tokens, 8,192 native TTS frames, Vivian, and unchanged realtime sampling.

| Turn | Spoken request | PCM bytes | Duration | WAV SHA-256 |
| --- | --- | ---: | ---: | --- |
| Solar | `详细介绍太阳系八大行星。` | 81,920 | 2.56 s | `205e8e77…a66` |
| Saturn interruption | `停一下直接讲土星环。` | 94,720 | 2.96 s | `7dceb7dc…de53` |
| English interruption | `换成英文讲。` | 38,400 | 1.20 s | `4b3cfa72…d218` |

The v1 Saturn rendering had a synthetic comma pause that split the interruption
into two VAD turns. V2 retains every spoken word and the original 2.96 s/94,720-byte
input span, replacing that internal pause with trailing digital silence. The failed
v1 profile and all launch-only failures remain in the external negative tree.

Playback is timed virtual PCM consumption. An ACK is sent only after the
corresponding samples have been consumed at 24 kHz. The first interruption follows
3.0 s of played solar narration; the second follows 2.0 s of played Saturn audio.
A zero text cursor is valid for the second response because playback is still in
its first native TTS leg; the driver does not extend the interruption window to
find a later cursor.

## Profile before candidate selection

The accepted Task #81 raw candidate journal decomposes its 10,760.964 ms result as
follows: cancellation at 3.756 ms, approximately 9.7 s of remaining paced input,
400 ms endpoint silence, about 111 ms from replay end to ASR final, and
532.621 ms from ASR final to new audio. The large value is not synchronous
history-store time.

The first successful Task #82 parent profile independently reproduced:

| Component | Milliseconds |
| --- | ---: |
| First barge → cancel | 4.354 |
| Cancel → digital-silence marker | 1.660 |
| First barge → new-segment speech stopped, including replay | 2,209.605 |
| Speech stopped → ASR final | 45.996 |
| ASR final → first new public audio | 622.497 |
| **Full first barge → new public audio** | **2,878.098** |

Replay alone exceeds 800 ms. Meeting the full gate would require responding before
the complete request, changing the frozen schedule, or moving the anchor. Those
options were rejected; no unrelated C++ candidate was attempted.

## Paired Before/After timing result

Three fresh-process pairs ran in the preregistered order
`before-r1 → after-r1 → before-r2 → after-r2 → before-r3 → after-r3` on explicitly
idle physical GPU 7. Quantiles use linear interpolation at `(n-1)*p`. All samples
and Tukey outliers are retained in the compact scorecard; there were **zero Tukey
outliers** in these metrics.

| First-interruption metric (ms) | Before median / p95 / max | After median / p95 / max | Gate |
| --- | --- | --- | --- |
| Barge → cancel | 4.393 / 4.407 / 4.409 | 4.375 / 4.404 / 4.407 | PASS <300 |
| Cancel → actual digital queue silence | 1.496 / 1.500 / 1.500 | 1.488 / 1.512 / 1.515 | PASS <300 |
| Barge → speech stopped, replay included | 2,209.882 / 2,210.073 / 2,210.094 | 2,209.832 / 2,210.344 / 2,210.401 | diagnostic |
| Speech stopped → ASR final | 46.235 / 46.280 / 46.285 | 46.289 / 46.420 / 46.434 | diagnostic |
| ASR final → first new audio | 622.324 / 622.348 / 622.351 | 622.380 / 622.398 / 622.400 | PASS <800 |
| **Replay-inclusive barge → new audio** | **2,878.448 / 2,878.511 / 2,878.518** | **2,878.391 / 2,878.620 / 2,878.646** | **NOT_MET <800** |

The second-interruption cancel median/p95/max is 1.653/1.679/1.682 ms Before and
1.715/1.717/1.717 ms After. Its full barge-to-new-audio median is 2,097.720 ms
Before and 2,097.788 ms After; this is also retained as a diagnostic. The roughly
0.056 ms first-full-path median difference is same-source noise, not a candidate
improvement.

## Exact PCM, history and lifecycle evidence

Every raw capture independently proves:

- Complete frozen audio append bytes and chunk-duration pacing, with no shortened
  or reordered recording;
- Exact played PCM prefix and final accepted cursor before each barge;
- Cancellation history equal to that exact ACK prefix, with canonical native VLM
  prompts matching the lossless per-turn history model;
- Pending old PCM is identified, hashed and cleared at cancellation; no old audio,
  text, ACK or terminal arrives after cancellation;
- Three expected response identities only, zero unexpected/cross-session response
  IDs, zero duplicate terminal callback kinds and zero late final-response events;
- Three exact native VLM prompts per journey, two retained truncated history turns,
  a final English-dominant Saturn-ring response with nonempty public PCM and one
  completed terminal, and final `in_flight=0`.

The first ACK cursor is 30 and the second is 0 in all six runs. The final English
text SHA-256 is
`8d5895fb62d924f07fc3541628120ea3db070c4fee729cc3c44a4c543542f12f`;
its complete 16,251,840-byte PCM SHA-256 is
`660363ddb78f3e0d7274e3c86e1a1a7f25759b832c0b5410f1aaf1a849d818e5`.

Pair 2 has 442 bytes of interrupted old text Before versus 448 bytes After because
cancellation catches a slightly longer generated prefix. Its played PCM, accepted
cursor, old PCM, pending PCM and VLM prompt history still match exactly. This is
retained as an `exact_pairing` failure rather than omitted. The final response text
and PCM match in every pair.

Audit replay is post-run independent raw decoding only. It is never sent as public
audio and never runs ahead of a new response.

## Required controls

| Control | Actual rerun result |
| --- | --- |
| `umm-test-local` | PASS, including five deterministic NS-04 replay/pairing fixtures; log `9991f049…3992` |
| Native TTS host + real CUDA determinism | 3/3 PASS; log `c61e82ca…ec2` |
| Public HTTP / delayed realtime | 25 HTTP responses, 24 complete + one exact cancel/retry; **29/29** responses and final ACKs, 493 exact PCM legs, 4,176 chunks, 115,034,880 bytes; exactly 4,125−29 = **4,096** pending audio events at release; verifier `4214b202…d2e4` |
| Realtime network/reconnect | Raw PASS: six sessions, three rounds, 15 connections, 41 connection-scoped identities, 30 exact next-turn prompts, zero stale/cross-session/duplicate/decreasing-cursor events, `in_flight=0`; verifier `f1c2737d…e94f` |
| Task #81 cursor-10 NS-04 | Independent raw PASS; exact played PCM `ccda9cf4…b003`, one 11,520-byte pending event, zero stale events; cancel 3.619 ms, silence 96.710 ms, post-ASR 532.568 ms, full replay 10,751.012 ms remains NOT_MET |
| Task #81 exact Chinese NS-01 lifecycle | **43/43 natural EOS**, 43 native terminals, 43 exact stream/terminal PCM legs and 43 exact native/public legs; clean text `883802d8…b572`, PCM `2ce2c534…4f78`; TTFA/image remain NOT_MET, I2V NOT_RUN |

The network rerun observed 27 terminal callbacks versus 32 in the historical Task
#81 summary. The raw independent verifier passes with zero duplicates and all
recovery invariants; no retry-to-preferred-count or verifier weakening was used.

The first NS-01 environment attempt failed before a public request because system
Python could not initialize CUDA. A compatible established PyTorch 2.6/CUDA 12.4
environment was selected. Its complete raw response then exceeded the generic
narrative postprocessor's assumptions by intentionally having no video, causing a
`StopIteration`. The accepted Task #81 NS-01-specific fresh verifier independently
redecoded that raw response and passed all 43-leg correctness checks. Both negative
postprocessing/environment attempts remain external evidence; full ideal NS-01 is
not claimed.

## Source, binary and model identity

Accepted parent: `15f54b98389ffecf0f9d100b0c9b4855975fe90d`.
Official main remained `91310b36184c86a1c147debb27db8643cc0cb3e4` at final
integration, so no newer merge was available. Paired live execution identity is
`310a9dd46f0bd893421ad83c07c9cf4dcfc21f2a`; final diagnostic/test source adds only
later verifier/scorecard corrections and documentation. The serving/native runtime
diff from the accepted parent is empty.

| Object | SHA-256 |
| --- | --- |
| `umm-infer` | `0f1850cb894896af2c5ded6678d559356eab9ae4a6353d4c454d36244d4e4562` |
| `umm-native-stage-asr` | `c28062aaace956e8a926fcf41fa86a4ea6c040fa543558bccc0776afbb62e834` |
| `umm-native-stage-qwen38` | `a3930bd382487c32315e3c869f1ab31e87c0c1a61515ab938cad85f15678f33b` |
| `umm-native-stage-tts` | `90fa74a114e96c621a7f36885a596f8f5b3442ddf1ea79098317c737f65cc86f` |
| Live harness | `2dcdfeb6db1866b93a95f2f02b532554cd6995688d176218e64405e5df5e0d94` |
| Independent verifier | `bf124a45c8369bd694f4d5a9ce42845881711d9fcce83b7522a8949186c8cdc1` |
| ASR model manifest | `a16807fc65e05f36687484bdfb8cb9f103c3fb59f6787d1f904f3ffebf8a3005` |
| Qwen3.8 model manifest | `fb2a8751df1dfbcdf556988acd8e2ac7c6f7180c609797d5e9b3a8e3e40a1c7e` |
| TTS model manifest | `92db72616c9578462add19ed81720b7abf1c1e9c3f4d225877a1cd6104d2fb27` |

CUDA 12.4.131 / SM90 / driver 535.161.08 was used. Model revisions remain
`UNKNOWN`; exact file manifests are hash-bound. Only physical GPU 7 ran CUDA work.
The resident service and GPUs 0–6 were not used or restarted.

## Independent verification and external evidence

Each of the six paired captures has two byte-identical fresh-process reports. A
further verification using only the bundled source and bundled input bytes is
byte-identical to the corresponding original report. The independent verifier does
not import the live producer; it decodes raw WebSocket frames and native MessagePack
submits, reconstructs exact PCM/history/terminal behavior, and recomputes all timing
and threshold decisions.

Raw media, input recordings, wire events, logs, negative attempts and complete
manifests remain in a caller-selected external evidence directory. The tracked
[compact scorecard](evidence/ns04-replay-lifecycle-sm90.json) contains compact
hashes/results only. Its SHA-256 is
`01d627ca3833455b7bc615538a9001de4be326675673d8078ec5b3bdce51801a`;
the controls projection is `15fb9433fd20c9a158f09259b27b7fd00686951486e298eb754d5dcedfc2d4c7`.
Final checks cover 228 feature-map rows / 1,164 references and 79 Markdown files /
276 local targets with zero missing links. The external `SHA256SUMS` excludes only
itself and fails on any missing, extra, unsafe or changed file.

## Remaining boundaries

- NS-04 replay-inclusive barge-to-new-audio remains **NOT_MET** and no score
  increase is claimed. The Task #83 [stable-complete incremental-response
  candidate](ns04-incremental-early-response.md) reduced the measured path only to
  about 2.718 s, retained an exact-pairing failure, and was removed from final
  runtime as NOT_PROMOTED.
- Physical microphone/speaker, acoustic echo and perceptual quality are **NOT_RUN**.
- Full two-GPU R4 was not rerun under the GPU-7-only constraint; accepted fair
  alternation and native model arithmetic are unchanged.
- No training, optimizer, deployment, resident-service change or video-speed
  improvement is claimed.
- The generic NS-01 narrative postprocessor still assumes a video artifact; the
  Task #81-specific raw verifier is required for the intentional no-video control.
  This harness limitation is not presented as a serving-model regression.
