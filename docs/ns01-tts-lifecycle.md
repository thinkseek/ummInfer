# NS-01/NS-04 native TTS and long-stream lifecycle (Task #81)

## Result and scope

**PROMOTED_CORRECTNESS; full NS-01 and NS-04 remain NOT_MET.** The exact frozen
Chinese black-hole request now produces complete deterministic native/public PCM
instead of leaving a nonterminal giant TTS tail. Barge-in, exact playback cursors,
cancellation, immediate retry, delayed readers, and network/reconnect regressions
pass. This is not a latency, video-speed, model-quality, deployment, physical-audio,
or perceptual-quality promotion.

The official runtime base is
`91310b36184c86a1c147debb27db8643cc0cb3e4`. Preregistration is
[preregistration.json](../benchmarks/ns01-tts-lifecycle/preregistration.json), committed
before candidate edits or new measurements as
`cdfaa9c2fececda486c2277e869a5dd2ede78892`. Runtime commits are
`55b8fc36887219a81d76810c609b9ddc81a8ebe8` and the independently reviewed
`5f8467ac28d36641a13bd6fe0608aa4b8f2361d1`; later harness/documentation commits do
not alter model compute. Task #79's TeaCache candidate and every video-speed
variant are absent. No training, optimizer, UI control, sampling default, native
Talker arithmetic, public schema, or fair-scheduling rule changed.

## Frozen exact Chinese case

The default public `POST /v1/chat/completions` streaming path uses autonomous
narrative routing, zero hand-authored visual directives, greedy seed 42, 768 maximum
VLM tokens, native 1.7B CustomVoice TTS at 24 kHz mono PCM16, and at most 8,192
codec frames per native leg. The user text begins with the exact NS-01 Chinese
rubric sentence and explicitly asks for a genuinely long multi-paragraph opening,
Markdown headings, and mixed `accretion disk`, `gravitational lensing`,
`event horizon`, and `Eddington limit` terminology. The complete frozen request is
hash-bound in the raw bundle; it was not shortened after observing failures.

Both arms produced identical 3,611-byte clean text SHA-256
`883802d889885bd032b18a3dc21c8f7e1ce77773f020ab0ee3e69a816ac8b572`.
That response exercised headings, emphasis split inside English words, multiple
paragraphs, and the 32-active-child boundary. It generated one fallback image but
no autonomous video slot. This visual/model-routing limitation is retained, not
rewritten into a successful I2V result. The preregistered opening-before-first-
heading criterion also remains unmet because the response starts with a heading.

## Root causes and focused correction

1. **Markdown was Talker input.** Retained Task #79 MessagePack frames prove that
   `**示意图：**` reached 8,192 frames / 655.36 seconds and `---` formed a
   Markdown-only request. The new parent independently left a multi-paragraph
   32nd leg nonterminal. Native speech input now removes supported presentation-
   only heading/list/quote, emphasis, code, and horizontal-rule delimiters.
   Single-star/underscore emphasis state survives span cuts. Ordinary `*` arithmetic
   and intraword underscores are preserved. Public Markdown and original UTF-8
   playback-ACK ranges are unchanged; markup-only spans advance their source cursor
   without creating a native request.
2. **The old 32nd leg coalesced all remaining text.** The existing 32 limit now
   bounds submitted-but-not-terminal native children. A native completion releases
   one slot and resumes the exact buffered text cursor. EOF no longer turns all
   remaining paragraphs into one oversized Talker request. No text or PCM is
   dropped to enforce the bound.
3. **Streamed chunks were charged to the terminal 60 MiB aggregate.** Each native
   PCM chunk is an independent immediately consumed artifact. It no longer
   exhausts the terminal image/audio/video aggregation budget. The 48 MiB per-chunk
   rejection and the nonstreamed 60 MiB aggregate rejection are unchanged. A
   deterministic 31 × 2 MiB test drains 62 MiB byte-for-byte with no abort.
4. **EOF could release a reserved partial directive.** An unterminated recognized
   `[[image:`/`[[video:`/`[[visual:` candidate is now finalized as removed directive
   bytes, not emitted into prose, history, or TTS. Literal non-directive brackets
   retain their existing behavior. The exact removed raw span is retained for
   fail-closed behavior-mask reconstruction.
5. **One long delayed reader could exceed 1,024 outbound frames.** The realtime
   drain now requeues the exact undelivered suffix in order instead of closing the
   reader. The existing 1,024 outbound and 4,096 unified-event bounds remain.
   No socket progress does not refresh the application response deadline. A
   completed ACK-mode response gets only a bounded final-ACK validity window;
   ordinary idle cleanup then resumes. The no-output 10-second stall and idle-only
   controls still pass.

Native `codec_frames` and `natural_eos` are log-only observations. They do not alter
model output or any public/native result schema.

## Exact public Before/After result

| Metric | Parent | Reviewed candidate | Result |
| --- | ---: | ---: | --- |
| Clean text bytes / exact pairing | 3,611 | 3,611 | identical SHA-256 |
| Native TTS submits | 32 | 43 | bounded continuation drained |
| Successful native terminals | 31 | 43 | candidate all complete |
| Nonterminal / aborted legs | 1 / 0 | 0 / 0 | corrected |
| Natural EOS observed | unavailable in old log | 43/43 | all intended legs |
| Native stream = terminal WAV PCM | 31/31 completed | 43/43 | exact |
| Native/public exact PCM legs | 32 prefix legs | 43/43 | exact |
| Public PCM chunks | 432 | 537 | no dropped suffix |
| Public PCM bytes | 12,199,680 | 14,987,520 | complete candidate |
| Public PCM SHA-256 | `a8c2e420…3cdc96` | `2ce2c534…4f78` | raw-bound |
| Active TTS high-water / final unsettled | 32 / 1 | 32 / 0 | bound preserved |
| Public terminal | cancelled | completed | one terminal + one DONE each |
| Total to terminal | 627,545.055 ms | 87,658.126 ms | parent timeout retained |
| TTFT | 560.983 ms | 600.873 ms | **PASS <800 ms** |
| TTFA | 27,562.247 ms | 29,421.350 ms | **NOT_MET <800 ms** |
| First image | 17,195.204 ms | 18,285.731 ms | **NOT_MET <2,000 ms** |
| I2V | absent | absent | **NOT_RUN / NOT_MET** |
| Tag/directive leaks | 0 in this parent | 0 | candidate clean |
| Native Markdown presentation leaks | 9 | 0 | direct raw-text check |

The candidate repeat has identical per-leg PCM, chunk counts, and tap-order native
aggregate `2ce2c534…4f78`. An intermediate analyzer briefly hashed lexicographic
filenames, which included different heartbeat frame names; that analyzer error was
rejected. The verified join always uses tap ordinal and native per-leg stream index.
No product nondeterminism is claimed from that discarded comparison.

The raw candidate's final cleanup assertion initially sampled 3 MiB / 1% and failed;
the raw log is retained. Owned processes had exited, and subsequent required checks
returned to 3 MiB / 0% with no compute process. The final all-gate cleanup is below.

## NS-04 paired long-narration engineering evidence

Both R2 arms used the same real 16 kHz PCM16 recording SHA-256
`0d44235340c7d9db54a03b8bdfa2fabc2f57b3dd90ab09ee638537b5650972c6`, a full ten-second
input allowance, 768 VLM tokens, 8,192 TTS frames, timed virtual PCM playback, and
three serial native stages on physical GPU 7. The ASR transcript asks for sixteen
repetitions. Both arms retained cursor 10 / `The small ` and played PCM SHA-256
`ccda9cf40688346b658929d145f0fc03ff050b4f4ce7eda3201f94b90a37b003`.

| Raw independently recomputed timing | Parent R2 | Candidate R2 | Hard result |
| --- | ---: | ---: | --- |
| First barge marker → cancel | 3.640 ms | 3.756 ms | PASS <300 ms |
| Cancel → digital-silence marker | 96.705 ms | 96.635 ms | PASS <300 ms |
| Post-barge ASR final → first new audio | 532.558 ms | 532.586 ms | PASS <800 ms |
| First barge → new audio, including real replay | 10,754.471 ms | 10,760.964 ms | **NOT_MET <800 ms** |
| Second barge marker → cancel | 1.401 ms | 1.343 ms | PASS <300 ms |

The parent retained one 11,520-byte pending PCM event; the candidate retained two /
42,240 bytes. Both clear pending virtual playback, immediately accept following
speech, produce a new audio response, and emit zero stale old-response events.
These are digital-loopback timings, not physical DAC/speaker measurements.

The first parent attempt's synchronous replay delayed raw cancel→digital-silence by
9,611.259 ms; that negative attempt remains in the bundle. The first candidate's
logic passed but postflight threw before writing its evidence JSON, leaving raw-only
incomplete evidence, also retained. R2 uses identical interruptible replay in both
arms and stable postflight rechecks; no old outlier is averaged away.

## Public PCM, network, and deterministic gates

The reviewed candidate passed the complete existing public HTTP + delayed-realtime
gate on GPU 7:

- 25 HTTP responses (24 complete + one cancellation), 44 legs, 184 chunks and
  10,698,240 bytes including the valid cancellation prefix;
- exact retry/repeat identity and the accepted 737,280-byte direct continuation
  control, preserving Task #59;
- **29/29** realtime responses, 493 exact native/public PCM legs, 4,176 chunks,
  115,034,880 bytes and 29 final playback ACKs;
- exactly 4,125 native audio events − 29 public frames = **4,096** pending events,
  all 29 readers paused, zero public reads during the pause, zero reset;
- verification SHA-256
  `13332021b24ee298c1955e2cf24c9ab26d3cc7ff7afc7cca18513b0acc0576a7`;
  nested manifest SHA-256
  `7a52825b36846e937b3b7383343dfe294017107e9b04b746dc28863d69973cb3`.

The existing six-session / three-round network gate also passed on GPU 7: 15
connections, 30 verified next-turn prompts, 41 connection-scoped response identities
(including ACK-only entries), 32 terminal callbacks, zero stale events, zero
cross-session leaks, zero duplicate terminal callback kinds, and final `in_flight=0`.
Physical playback and listening remain NOT_COVERED.

Deterministic red/green records cover Markdown/cursor finalization, 62 MiB streamed
PCM, 40 sentences with a 32-active high-water mark, EOF directive suppression,
1,025 delayed realistic audio frames, exact final ACK, cancellation/retry, and stale
suppression. The original no-progress application timeout remains enforced.

| Gate | Result / raw SHA-256 |
| --- | --- |
| Final `umm-test-local` | PASS; `fd5b84bb68151e58d375b28a28e14b8eacecc8549a0bcfdae7bc1651d029d5e8` |
| Reviewed unified regression run | PASS; `30e48c79e901a27b5515d8dcf60c2fe85baa92c25075a6f5ea4b07b0db6ac140` |
| Public PCM deterministic tests | PASS; `d36364a9815f4dce4c5147229e55f354b472ce6b6433c593bb5c2f11a51ec45b` |
| Realtime network deterministic tests | PASS; `ce6cfbbe582758a5f9035e376902e6059d9fa33e5dc3c7c82d7fadbc2c25e929` |
| Native TTS host + real CUDA determinism | 3/3 PASS; `fe1948eab21f9f351a5b297f12f6a242cd765464ce961be4f27378c102b8c6d1` |
| Parent active-bound/Markdown/EOF red run | expected FAIL; `6ccb86afee6adfadd3d4eba5c455222bdf531be062d0c2df0d73115439527d71` |
| Parent 1,025-realistic-chunk delayed red run | expected FAIL; `835ad71e3219ca37740bbb92bc7da728bad507639234c774430897c97a43a7ef` |

A first tiny-payload 1,025-chunk control did not expose socket pressure and stayed
green; it was not used as the red proof. The realistic-payload control failed before
the requeue correction and passes afterward without dropping byte/terminal checks.

## Source, binary, model, and configuration identity

All real runs used only physical GPU 7 after explicit idle checks. The reviewed
candidate binary SHA-256 values are:

| Binary | SHA-256 |
| --- | --- |
| `umm-infer` | `0f1850cb894896af2c5ded6678d559356eab9ae4a6353d4c454d36244d4e4562` |
| `umm-native-stage-qwen38` | `a3930bd382487c32315e3c869f1ab31e87c0c1a61515ab938cad85f15678f33b` |
| `umm-native-stage-tts` | `90fa74a114e96c621a7f36885a596f8f5b3442ddf1ea79098317c737f65cc86f` |
| `umm-native-stage-asr` | `c28062aaace956e8a926fcf41fa86a4ea6c040fa543558bccc0776afbb62e834` |
| `umm-native-stage-video` | `d1e5e4f8bf9e967b3060d7633de498998c0b2a7936da405368c224c7048300db` |
| `tts_cuda_determinism_test` | `7b26d98a5dbe53ef15165acf85c91f3fcd0ddc73334e18f4499243c7317062fb` |

Model revisions remain UNKNOWN; exact file manifests and configuration hashes are
sealed. Tree SHA-256 values are Qwen3.8
`22e638404d0e413007064c47d5c98ad1696a26a0e980eb2fce83cbbfcca0a304`,
Qwen3-TTS `69c90c52d9d88390f4b4f540b09acf40f5ead7080650585ff20d2e144aa5e69d`,
FLUX.2 `10c1b2d7096cd6c2798fa9d2b9aecfab5163117cbf3d66de6f32f9a2e8f17997`,
and Wan2.2 `b118ac07b533cf7d68a4806ec70bdf39041515673e44535afeda5cfefe761616`.
CUDA 12.4 / SM90 was used for native CUDA tests. No other physical GPU was used for
model execution, and the resident service/GPUs 2–6 were never contacted or restarted.

## Independent verification and retained evidence

The compact machine-readable result is
[ns01-tts-lifecycle-sm90.json](evidence/ns01-tts-lifecycle-sm90.json). It contains
raw response/PCM hashes, every per-leg outcome, all hard-threshold failures, paired
NS-04 values, public/network results, and retained negative-tree identities. Raw
media, per-event payloads, model files, and absolute evidence paths are not tracked.

The complete external bundle has **4,774 files** and a self-excluded `SHA256SUMS`
SHA-256 `44afa02c8d1227b5dbf00197186eda67051ef66226a1db07333912cc832d894d`.
Scorecard SHA-256 is
`9a4050a6f4649b265f1b5806384367c8e4eb09bd69083af60dbb1f2277d2d7d6`.
Two fresh independent `tests/ns01_tts_lifecycle_verify.py` processes produced
byte-identical PASS reports SHA-256
`d8c61d81761325eec21f9a3755aaa13edbca1d5839ca0800e2f2d3e8f1f64932`.
They do not import the producer. They redecode raw SSE/MessagePack/WAV and NS-04
client/server journals, recompute every PCM/EOS/timing/promotion decision, and verify
all nested public/network manifests and negative-attempt trees. Complete raw evidence
stays outside Git.

Final GPU 7 state is **3 MiB / 0%, zero compute processes**; all task-owned processes
exited and final `in_flight=0`. No deployment occurred.

## Remaining boundaries

- NS-01 TTFA, first-image, and missing-I2V gates remain NOT_MET; no NS-01 score
  increase is claimed.
- NS-04 replay-inclusive barge→new-audio remains NOT_MET; physical microphone/speaker,
  echo/perceptual listening, and acoustic quality are NOT_RUN. Task #82's paired
  [replay/lifecycle diagnostic](ns04-replay-lifecycle.md) preserves that boundary and
  is NOT_PROMOTED with zero serving/native runtime change. Task #83's bounded
  [incremental-response candidate](ns04-incremental-early-response.md) is also
  NOT_PROMOTED and absent from final runtime; the accepted Task #81 controls remain
  the correctness baseline.
- The supported Markdown normalization is deliberately bounded to the tested
  presentation forms, not a claim to be a complete Markdown parser.
- Full two-GPU R4 was not rerun under the GPU-7-only constraint. Task #73's accepted
  serial fair-alternation code is unchanged; public PCM and real network regressions
  passed instead.
- Unseen-prompt generalization, production deployment, training, optimizer, and any
  video-speed improvement remain NOT_RUN or unclaimed.

## Reproduction

Use exact local assets, a fresh caller-selected output directory, and an explicitly
idle authorized GPU. No tool has an absolute output default. The principal entry
points are:

```sh
UMMINFER_LIVE_OPT_IN=1 /path/to/python -I \
  tests/public_narrative_i2v_live.py --scenario ns01-chinese-markdown-long \
  --out-dir "<new-evidence>" --gpu 7 \
  --server-binary /path/to/umm-infer \
  --vlm-binary /path/to/umm-native-stage-qwen38 \
  --tts-binary /path/to/umm-native-stage-tts \
  --wan-binary /path/to/umm-native-stage-video \
  --flux-stage python/umminfer_flux2_stage.py \
  --python /path/to/python --vlm-model /path/to/qwen3.8 \
  --flux-model /path/to/FLUX.2 --tts-model /path/to/Qwen3-TTS \
  --wan-model /path/to/Wan2.2 \
  --base-commit 91310b36184c86a1c147debb27db8643cc0cb3e4 \
  --execution-commit 5f8467ac28d36641a13bd6fe0608aa4b8f2361d1
```

`tests/ns01_tts_lifecycle_scorecard.py` binds the raw arms and optional NS-04,
public, network, log, and negative directories. Run
`tests/ns01_tts_lifecycle_verify.py <bundle> --out <new-report>` in a fresh process;
its output must remain outside the sealed bundle.
