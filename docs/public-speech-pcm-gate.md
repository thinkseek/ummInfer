# Public HTTP speech to native-PCM parity gate (Task #63)

## Scope and execution identity

This gate continues Task #59 without changing native TTS kernels, scheduling,
public schemas, online training, image/I2V, or video selection. The integrated
source is official Wave 91 `ced4b145`, which preserves Task #66's held-out
video records, Task #64's blocked optimizer evidence, Task #65's public
narrative/cancellation records, Task #62's native I2V continuity records,
Task #61's temporal motion-run records, and Task #59's repetition-penalty fix.
The final harness commit is `50adb075011adc2a99a518e8239274304818c562`.
Two timeout experiments were reverted after they did not prevent the resets;
the final tree has no serving change. No scheduler rule, public schema, native
kernel, or public PCM chunk size changes.

The live topology is one isolated H20 (SM90) physical GPU 0, one loopback
gateway, and serial native ASR/Qwen3.8-27B/Qwen3-TTS-1.7B-CustomVoice stage
processes. HTTP chat uses only the VLM and TTS entries. The smaller realtime
block uses ASR only to exercise the existing server playback-acknowledgement
and delayed-reader contracts; it is not the full ASR/two-GPU R4 replay. No GPU
1/7, other resident GPU, or resident service is used. Physical playback and
perceptual listening are outside the claim.

After the HTTP matrix completes, the task-owned initial TTS process is stopped
cleanly. Realtime ASR/VLM then runs while the same pinned TTS binary is restarted
with a 120-second delayed re-registration. Coordinator holds TTS submissions
during that existing readiness window, allowing public VLM text to drain before
native audio begins and keeping the delayed-reader gate inside the unchanged
600-second response deadline. Initial/restarted process identities are retained.
This is the repository's existing stage-hot-restart contract, not a scheduler,
timeout, threshold, request-schema, or native-kernel change.

| Object | SHA-256 |
| --- | --- |
| `umm-infer` | `bfee9ff8e9bb111f0ae48dd97ed596c46340749dc15f03d31258138ed1ac0f5c` |
| `umm-native-stage-asr` | `688de7b55b551bed365ae9ee57140b6317a282faf33c2520ac1ce8b15180ebf9` |
| `umm-native-stage-qwen38` | `38294a513d1ea5d26592ad871fdcef808a86606923c6181c4e19a80770c68b8a` |
| `umm-native-stage-tts` | `978e29069888438e84242009d95e83ef51d2bef5fbaa1fadb0bd9cad4c49f069` |
| ASR model manifest | `a16807fc65e05f36687484bdfb8cb9f103c3fb59f6787d1f904f3ffebf8a3005` |
| VLM model manifest | `fb2a8751df1dfbcdf556988acd8e2ac7c6f7180c609797d5e9b3a8e3e40a1c7e` |
| TTS model manifest | `92db72616c9578462add19ed81720b7abf1c1e9c3f4d225877a1cd6104d2fb27` |
| ASR `config.json` | `2e74a751548b8ad7d7526d29365ad8144c345d8b412b1152d25dc6698452712f` |
| VLM `config.json` | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| TTS `config.json` | `17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9` |
| `native/tts/cuda_kernels.cu` | `18e8ccf5dddcdee1968472811f22c72b854ea448c0d4c98a13486fe303c1079e` |
| `tests/public_speech_pcm_live.py` | `529a04517aad6a93a1e6aba51c56786b56429e38613ff68802d64eb3815a542f` |
| `tests/public_speech_pcm_verify.py` | `f0778a4a1929a6937dc4c06312aefb1a023c89a93a98be368b88738260e30cd6` |
| `tests/public_speech_pcm_test.py` | `eb3585d3469a348594ae5620756e4fcb493f49f9ba1b395c3ae1e6c290f3afe9` |

The realtime instruction is real native-TTS audio, resampled to the established
16 kHz mono PCM16 ASR input contract. Its input SHA-256 is
`0d44235340c7d9db54a03b8bdfa2fabc2f57b3dd90ab09ee638537b5650972c6`.
It asks for sixteen repetitions of one sentence. A one-response probe retained
an exact full ASR transcript, sixteen VLM sentences, 144 public PCM chunks, and
3,966,720 PCM bytes with a valid final ACK. That probe is a topology check, not
the final concurrency result.

## Gate matrix

`tests/public_speech_pcm_live.py` owns every gateway/stage process and places
byte-preserving forwarding observers on ASR/VLM/TTS submit and native
completion edges. Raw observer frames include original MessagePack bytes,
SHA-256, and monotonic receive/forward timestamps. HTTP captures retain request
JSON, raw response bytes, decoded bodies, seeded `recv()` sizes (including
timeouts), and delayed-read/abort records.

The HTTP matrix has 25 responses:

- short opening and multi-sentence continuation prompts;
- SSE sentence-streamed PCM, SSE one-shot WAV, and nonstream JSON WAV, each with
  an initial response and two fresh repeats;
- one additional delayed HTTP reader, one real cancellation, and one retry;
- three fresh direct `/v1/audio/speech` PCM responses and one direct WAV response
  for the exact Task #59 continuation text.

The VLM continuation is retained exactly as generated, including its `you'r`
spelling. It is never substituted for the direct control's exact `you're` text.
The direct control must reproduce 737,280-byte PCM SHA-256
`3eaa2791bba77428558a314086e2264e3279123b03d89a4998f24a3cce11773d` on every
PCM/WAV payload. This proves that the accepted repetition-penalty fix is loaded
without rewriting different VLM prose to fit a hash.

The realtime block uses 29 fresh sessions. A reader pauses if its first
`audio.delta` arrives before the release. Every currently paused reader resumes
only when at least one reader is paused and the retained live counts satisfy:

```text
native TTS audio chunks - public audio.delta frames >= 4096
```

VLM text, ASR events, terminals, ACKs, and other control events do not count.
Unpaused readers continue draining, avoiding a self-deadlock in which later
VLM/text traffic prevents their first audio from arriving. Paused readers send
existing WebSocket PING frames every 30 seconds from independent heartbeat
threads. PING/PONG does not count non-audio events and does not change the
600-second response deadline; the retained resets still occurred despite these
heartbeats. Release records bind the exact paused response-id set; later
first-audio readers do not create
artificial pause records. Failure to prove the audio-only backlog
before the 1,800-second timeout is a hard failure. After release, all 29
responses must deliver every text/PCM chunk and final playback ACK exactly once
with correct terminal state and no stale event.

## Independent verification

`tests/public_speech_pcm_verify.py` runs twice in separate fresh processes and
never imports the live driver. It independently:

1. decodes raw chunked/non-chunked HTTP, SSE, masked/fragmented WebSocket, and
   MessagePack records;
2. checks the seeded read-size schedule, raw hashes, gap-free event sequences,
   one terminal, and no stale public event after terminal;
3. binds native VLM text using the existing parent or `:vlm` compatibility id;
4. binds every public PCM chunk to consecutive native TTS chunk indices, then
   checks UTF-8 cursor bounds and every inter-leg text gap against the actual
   native submitted sentence;
5. compares stream/terminal PCM, SSE/nonstream JSON WAV, direct PCM/WAV, all
   fresh repeats, cancellation prefixes, and retry identity;
6. requires accepted ACKs to reference already-issued nondecreasing cursors,
   rejects foreign ACKs, and requires one final server `response.playback_ack`
   per realtime response at the full-text cursor; and
7. requires `in_flight=0`, owned-process exit, and GPU 0 cleanup before writing
   a PASS report.

Successful responses require exact native/public VLM text. A cancelled response
may retain legal late native text, but its public text must be an exact prefix
and no late public event is allowed. Abort frames on submit edges are retained
as control records, not misclassified as work submissions.

`tests/public_speech_pcm_test.py` provides 19 deterministic regressions for
missing/reordered/stale chunks, byte tampering, UTF-8 cursors, terminal gaps,
seeded HTTP/WebSocket fragmentation, VLM-id ambiguity, abort filtering,
cancellation text prefixes, pausing-constructor installation, and fail-closed
audio backlog release. `umm-test-local` registers these as
`public-speech-pcm-test` in the full native/Python suite.

## HTTP rehearsal result (before the final realtime block)

The retained 25-response HTTP rehearsal passed independent fresh-process
verification: 24 completed and one cancelled response, 44 PCM legs, 184
streamed chunks, and 10,698,240 public PCM bytes including the cancel prefix.
Every repeat group matched, the cancel retained one valid PCM chunk, and retry
matched the fresh continuation. Four exact direct-control responses reproduced
the accepted 737,280-byte hash. The diagnostic verification report SHA-256 is
`e7fbb1304f14a6966ac20f762853bc5687e9b50e59ae374b02c1726f7da25720`.

Earlier failed attempts remain raw evidence: a separated-binary launch error
(no GPU launch), a nonstream header-timeout harness error, an abort-control
verifier error, a two-second instruction-truncation error, an interrupted
concurrency attempt whose pausing constructor was not installed, and an
all-readers-paused deadlock that retained 26 paused readers, 3,744 native audio
chunks, 26 public audio frames, and only 3,718 pending audio events. None is
counted as a passing gate. The constructor defect and the active-paused-reader
release rule were fixed and covered by deterministic regressions before the
final attempt; the 4,096 audio threshold was never lowered. A later attempt did
prove 4,125 native minus 29 public audio events but exceeded the unchanged
600-second response deadline before final drain. Longer/digit instruction probes
were rejected when ASR split or omitted text and are not used as evidence.

## Final outcome: blocked realtime completion

The final no-serving-change capture proves the required audio-only backlog:
**4,125 native TTS chunks minus 29 public `audio.delta` frames = 4,096**, with
all 29 paused response ids retained. Only **17 of 29 realtime responses**
completed with valid final playback ACKs; **12 reset** before terminal. Thus
public HTTP is **PASS**, the delayed-reader backlog is **REACHED**, and full
realtime parity is **BLOCKED**, not passed. No final `verification-r1.json` or
complete `SHA256SUMS` was generated for the failed run, and no such PASS is
claimed. The timeout experiments were excluded from the final source.

## Limits

Full ASR/two-GPU R4 and fair-priority reevaluation are **NOT_RUN**. Physical
microphone/speaker playback and perceptual/acoustic quality are **NOT_RUN**.
The realtime acknowledgement is a digital PCM-consumption protocol result, not
proof that a physical speaker played audio. No latency improvement, production
deployment, unseen-prompt determinism, or generalized model-quality claim is
made.

## Task #70 realtime transport-lifecycle correction

The lifecycle invariant and pass criteria were preregistered before serving edits
(preregistration SHA-256
`5a731c56aee7857c27cc2aef625c8bb916be8c86bd9ee0bd6a5a8211cf197293`).
Execution commit `2b49b5b82739eb0055ae5737595f693b08650b18` merges official
Wave 94. The retained heartbeat-enabled Task #63 capture shows that the 12
reset connections were the same 12 libwebsockets sessions destroyed about 5.25
minutes after establishment. Its 300/310-second validity policy confirms only
PONG (or an explicit validity refresh), not the client's periodic PING. Delayed
application reads could therefore be misclassified as an idle transport even
while their ASR/VLM/TTS response remained active.

`RealtimeApi` now names the production 300/310 policy explicitly and refreshes
connection validity only while a session owns active ASR/response work, or while
a generation-complete response still has queued output. It does not modify
`response_deadline`, `request_timeout_s`, any queue bound, PCM chunking, cursor,
terminal, ACK, scheduler, or native TTS behavior. Ordinary idle expiry resumes
once that ownership ends.

The real-loopback deterministic regression used PING=1 s / hangup=2 s and no
client PONG. Pre-fix, active work was destroyed at 2.001 s, `response.completed`
was lost, and the 10 s response-stall control also failed at 2.001 s (red log
SHA-256 `721014451b46cc97c36283045df0cf8b4c26d28f326c8ec74b7b3d40f775214a`).
Post-fix, exact text/PCM and one final ACK completed, the true stall still ended
at 10 s, and an idle-only connection closed (green log SHA-256
`db613f8a252d30986417a7091481f4e64cab0a64a906910b8730fe20458e2571`).
`umm-test-local` passed, including the complete unified contract binary and 21
public-PCM regressions. The live driver now runs exactly one measured response
per session, while independent verification requires all 29 paused readers and
exactly 4,096—not merely at least 4,096—native-minus-public audio events.

## Task #70 real SM90 outcome: PASS

Execution commit `2b49b5b82739eb0055ae5737595f693b08650b18` merges official
Wave 94 `6e32233f`, including Task #67, Task #63, and Task #69. An earlier R1
launcher was empty and exited without creating a capture or using the GPU; it is
not counted as evidence. R2 used a hash-checked launcher and fresh binaries on
explicitly idle physical GPU 0 only.

All 25 public HTTP responses retained their prior controls: 24 completed and one
cancelled, 44 PCM legs, 184 chunks, 10,698,240 bytes including the cancel prefix,
one-chunk cancellation, exact retry, and four direct-control responses with the
accepted 737,280-byte continuation hash. The delayed-realtime block then
completed **all 29 responses with one terminal and one final playback ACK each**.
Independent per-response/per-leg reconstruction checked **493 TTS legs, 4,176 PCM
chunks, and 115,034,880 bytes** with exact native/public joins, monotonic
cursors, and zero stale events or turn failures.

The release retained all 29 paused response ids and was exactly **4,125 native
TTS audio events minus 29 public `audio.delta` frames = 4,096**. During the
paused windows there were 4,096 native audio events and zero public audio reads.
No audio was dropped, reordered, or coalesced, and no terminal, ACK, cursor, PCM,
queue, or timeout assertion was weakened.

Two fresh verifier processes produced byte-identical 1,691,984-byte reports with
SHA-256 `cda99830359798b3a339601babb4c22341d39f582b21bbcb68414f4fe167c716`.
The expanded raw bundle includes the exact instruction WAV, source, driver and
red/green/build/test records. Two fresh manifest checks covered **1,266 files**
with self-excluded `SHA256SUMS` SHA-256
`3adbd3d50dd4b6a825ce77943a35ba3da4f54c15a1316913692f950ddc1bbb31`.
`umm-test-local`, all 21 focused PCM tests, the 223-row/1,125-reference feature
map, and all three native TTS tests passed. `in_flight=0`, every task-owned
process exited, and GPU 0 returned to 3 MiB / 0% with no compute process; the
resident service and its devices were untouched. Machine-readable identities and
limits are in [realtime-transport-validity-sm90.json](evidence/realtime-transport-validity-sm90.json).
At that Task #70 checkpoint, full two-GPU R4, fair-priority, deployment, physical
playback, and perceptual quality remained **NOT_RUN**; no latency improvement was
claimed.

## Task #73 final candidate regression: PASS

The Task #73 fair-alternation candidate reran this complete public gate on its
exact fair binaries after the two-GPU R4 and real network gates passed. HTTP again
passed all 25 controls (24 completed + one cancelled), including exact retry and
four direct 737,280-byte continuation controls. All **29/29** delayed-realtime
responses completed with one terminal/final ACK and exact native/public PCM:
493 legs, 4,176 chunks, 115,034,880 bytes. Release remained exactly
4,125−29=4,096 with every reader paused and zero public reads during the pause.

Three remote plus two isolated-local fresh verifier reports are byte-identical at
`fa24b43d78d351bd209566b60a37ba2cba0b7504296d0dfe2f6c8b02a045023a`; the
1,251-file self-excluded manifest SHA-256 is
`b1f2f34fd7cd0e0d865ddcd65b16677d143d773bf21bc9472946315d5beafff5`.
All owned processes exited, `in_flight=0`, and GPU 0 returned to 3 MiB / 0% with
no compute process. The public verifier's `full_asr_two_gpu_r4=NOT_RUN` field is
capture-local and remains unmodified; Task #73's actual R4 result is recorded
separately in [realtime-r4-post-determinism](realtime-r4-post-determinism.md),
along with network/reconnect evidence, exact hashes, and unchanged physical /
perceptual limitations.
