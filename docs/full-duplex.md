# Full duplex, L1: streaming input, server endpointing, interruption

This page defines the first full-duplex layer, records what was measured on
SM90, and lists what L2 and L3 still need. It does not claim any behavior
that was not run.

## Definition of L1

A client streams PCM16 to the realtime WebSocket while the server transcribes.
The server, not the client, decides when a turn ended, and a new utterance
stops the previous turn's computation at a computation boundary.

L1 is complete when all five statements hold:

1. Audio appended in chunks reaches the ASR stage before `transcription.done`;
   the first provisional transcript arrives in less than one second for a real
   utterance.
2. `turn_detection` accepts a server configuration. Digital silence never
   starts a turn.
3. A pause longer than `silence_ms` commits the turn without
   `transcription.done`.
4. A new utterance while an earlier request computes aborts that request, and
   its output stops within one second.
5. The local suites and the real native ASR path pass.

## What the server does now

`src/serve/realtime_api.cpp`:

- **Incremental submission.** Every 200 ms (`kPreviewWindowBytes`, 6400 bytes)
  of new audio starts a preview request over the turn's audio so far, if no
  request is in flight. A slower stage coalesces windows instead of queueing
  them. Preview requests use the id space `realtime-preview-N`; turn requests
  keep `realtime-N`.
- **Cumulative previews.** Each completed preview emits one
  `transcription.segment` with `event_index: 0`, the turn's `segment_id`, the
  cumulative text and `is_final: false`. A preview failure is dropped and does
  not fail the turn; the next window retries.
- **Server VAD.** `turn_detection` accepts `null` (disabled, the previous
  behavior) or `{"type": "server_vad", "silence_ms": N}` with N in
  `[100, 5000]`, default 500. `src/serve/energy_vad.hpp` is a 10 ms-frame
  energy detector with an adaptive noise floor. `echo_robust_vad.hpp` wraps it
  with the bounded playback-reference residual gate described below; neither
  component adds a dependency.
- **Endpointing.** `input_audio_buffer.speech_started` and
  `input_audio_buffer.speech_stopped` are emitted on VAD transitions. The stop
  event submits the turn's final request without `transcription.done`. Under
  server VAD the WebSocket stays open after `transcription.completed`, and the
  next utterance becomes a new turn with `segment_id + 1`.
- **Interruption (barge-in).** A VAD speech start while a request is in flight
  calls `Coordinator::abort`, releases the record, clears the provisional
  segment for the client, and starts the new turn at the onset frame. The
  stage stops at its existing abort check: the native ASR engine checks every
  audio-tower layer, every prefill layer and every generated token, and the
  Python stage checks between yields.
- **Compatibility.** Without `turn_detection`, `transcription.done` still
  submits the whole turn and the socket still closes after the result. The CLI
  client is unchanged.

The native ASR engine already had an abort hook (`native/umm_asr_native/engine.hpp`,
`AbortCheck`), so L1 uses it; no new checkpoint was needed in the stage.

## Measured on SM90, 2026-11-17

Environment: SM90 GPU container, native ASR on GPU 2, local Qwen3-ASR model
directory (`$UMM_ASR_MODEL`), real speech `asr_en_pcm16.wav`, server and
stage built from this commit. The server used ephemeral ports; the resident
services on 43300 and 43301 and GPU 0, 1, 4, 5 were not touched.

Command, inside the container, using the committed script
`tests/full_duplex_live.py` (supply your real `--bin-dir`, `--model`, `--audio`,
`--out`, and `--host` / `UMM_DUPLEX_HOST`):

```sh
python3 -I tests/full_duplex_live.py \
  --bin-dir "$UMM_BIN_DIR" \
  --model "$UMM_ASR_MODEL" \
  --audio "$UMM_ASR_AUDIO" \
  --out "$UMM_OUT_DIR/evidence-repo" --gpu 2 --host "${UMM_DUPLEX_HOST:-127.0.0.1}"
```

Raw evidence: `$UMM_OUT_DIR/evidence-repo/live-duplex-evidence.json`,
`server.log`, `stage.log`. It is not a committed artifact. The script exits 0
only when every check below passes and the barge-in interrupt was observed.

| Check | Result |
| --- | --- |
| `session.update` echo | `{"turn_detection":{"type":"server_vad","silence_ms":400}}` |
| Silence does not trigger | 500 ms lead-in plus 500 ms wait: 0 events, 0 turns |
| First provisional transcript | 601.4 ms after the first speech chunk (2 s of speech sent in 100 ms chunks at 100 ms pacing); text `嗯。` |
| Server-side first preview | `realtime first_partial_ms=1601 segment=0`, measured from the first byte of the session, so it includes the 1.0 s of leading silence |
| Endpoint commit | 559.5 ms from the first 100 ms silence chunk to `transcription.completed` (400 ms `silence_ms`, 100 ms pacing, the rest is final compute) |
| Preview cadence | 9 previews for a 2 s turn; stage compute 149-219 ms for 0.5-1.4 s of audio |
| Barge-in interrupt latency | New speech sent at `t_barge`; server logged `realtime interrupt reason=barge_in request=realtime-2 epoch_ms=...` 201 ms later |
| Abort to stage exit | Stage logged `aborted at token 0 ... abort_epoch_ms=... exit_epoch_ms=...`, 19 ms later (24 ms in an earlier run) |
| Old output after the barge-in | 0 segments and 0 `transcription.completed` from the aborted turn (`segment_id` 1) |
| New turn after the barge-in | First preview (`segment_id` 2) at 430 ms, first non-empty text and `transcription.completed` at 984 ms, text `嗯。` |

The 201 ms interrupt delay is two 100 ms chunk boundaries plus scheduling: the
VAD can only see speech once a chunk carrying it arrives.

Residuals observed in the same run:

- The first previews of a new turn can carry empty text (430/560/701 ms)
  because the ASR stage needs more than a short prefix.
- The aborted turn's provisional text stays on screen until the new turn's
  first preview replaces it; the server sends an empty provisional segment at
  the barge-in, and the client replaces it at 430 ms.
- The bundled CLI client treats the first `transcription.completed` as
  terminal. Multi-turn reuse of one WebSocket is a server capability that the
  CLI does not yet exercise.

## Known limits of L1

- Energy VAD only. Steady noise above the adaptive floor by 6 dB is classified
  as speech. The floor starts at -60 dBFS and learns only from non-speech
  frames.
- One request per session at a time. Previews re-encode the whole turn prefix,
  so cost grows with the square of turn length. The 200 ms window is a floor on
  latency, not a throughput claim.
- Echo suppression is a bounded correlation/residual slice, not a full acoustic
  echo canceller. An acoustic path outside the 500 ms search window, an
  unobserved or poorly aligned playback path, and loud non-echo noise fall back
  to energy VAD and can still start a turn. Physical-device behavior is
  `NOT_RUN`; see the required live gate below.
- The residual is used only for VAD. ASR still receives the original microphone
  PCM, so no double-talk transcription-quality claim is made.
- The preview window is not configurable on the wire.

## L2: streaming output, sentence pipeline, prefix reuse

L2 closes the two product gaps that L1 left: the answer waited for the whole
text before audio started, and every turn re-prefilled the whole prompt.

### Streaming TTS

`TtsEngine::synthesize` takes an optional `StreamSink`. The talker emits its
first chunk after 2 codec frames (160 ms of audio) and later chunks every 8
frames (640 ms). A chunk decodes the frame prefix generated so far; only the
samples past the frames already emitted are kept, and they are appended to the
one vector the engine returns. The codec is causal, so a prefix decode has the
same mathematical value as the head of the full decode.

The TTS stage runs this path when the request sets `stream=true` and emits one
stage-stream chunk per codec boundary:
`{"audio": <PCM16>, "index": i, "first_frame": f, "frames": n,
"sample_rate": 24000, "channels": 1, "format": "pcm16"}`. The terminal result
still carries the complete WAV and the phase times.

Within one streamed request the concatenated chunks and the terminal WAV come
from the same buffer, so they are equal bit for bit. The stage checks this on
every streamed request (`tts stream_check identical=1`) and raises a protocol
error if it ever fails, so a regression cannot pass silently.

Across requests the F32 codec is not bit-repeatable. Two identical CLI runs of
the same text produce waveforms that first differ at sample 19422 with a
maximum difference of 9 PCM16 units (2.7e-4 of full scale); a repeat decode of
the same frames inside one process also differs (`tts repeat_check`). The
implementation therefore decodes each frame prefix once and never replays a
decode. No cross-run bit equality is claimed, and no listening test was run.

### Sentence pipeline

`include/umminfer/pipeline/sentence_split.hpp` cuts the VLM delta stream:

- strong terminators end a sentence: `。 ！ ？ ! ? ； ;`, newline and `…`;
- closing quotes/brackets and repeated terminators stay with the sentence;
- `.` ends a sentence only when the next codepoint is whitespace, end of text
  or CJK, and the previous codepoint is not a digit, so `3.14` stays whole;
- weak breaks are `， , 、 : ：` and space;
- with no terminator, a 48-codepoint window cuts at a weak break in its second
  half, otherwise at the window edge;
- the first TTS leg may cut at a weak break after 8 codepoints, so a long
  opening sentence does not delay all audio;
- a cut never splits a UTF-8 sequence.

Uncertain boundaries are not handled: abbreviations (`Mr. Smith`, `e.g.`) cut
after the period, `U.S. Army` becomes `U.S.` plus ` Army`, and a comma inside a
number is a weak break only at the length window. This is punctuation, not a
semantic end-of-turn model (that is L3).

The unified engine submits one TTS child leg per cut, `<id>:tts:0`,
`<id>:tts:1`, ... in answer order. A non-streaming request keeps the frozen
single `<id>:tts` leg and the one-shot artifact. When `audio.stream=true` is
requested alongside image/video or interleaved narrative output (Wave 10+),
sentence-level PCM16 streaming runs concurrently with visual legs; non-streaming
mixed requests keep the single post-VLM WAV leg. The request completes only when
the VLM leg is terminal and every output leg is terminal.

Each streamed chunk becomes one `artifact` event with `media_type: "audio"`,
`format: "pcm16"`, `mime_type: "audio/pcm"`, `sample_rate: 24000`,
`channels: 1` and `index: 0, total: 1`. The SSE frame carries the PCM16 base64.
The terminal WAV is not re-emitted.

### Realtime spoken response

`session.update` accepts `audio_output: {"voice": "vivian"}`. It needs a
unified engine with a registered TTS stage and server VAD; otherwise the
server rejects it. After `transcription.completed`, the listener submits a
VLM+TTS request for the transcript and forwards its events on the same socket:

- `response.text.delta` (`text`);
- `audio.delta` (`audio` base64 PCM16, `sample_rate`, `channels`, `format`);
- `audio.done`;
- `response.completed` or `response.failed`.

A new utterance cancels the response request through the unified engine, so
barge-in stops both the VLM and the TTS stage at their existing checkpoints.
Cancellation also works after the ASR request id has been cleared and after
generation has completed but playback is still in progress. Queued events for
the old response id are purged and late events are dropped.
When the server is started with `--session-history`, realtime responses opt into
`session_history=true`. With the default compatibility configuration, completed
turns keep the existing history behavior and cancelled/failed turns stay
excluded. The opt-in playback-acknowledgement contract below adds the narrow
realtime barge-in exception. The session id also keys the VLM prefill cache.

The turn buffer now drops exactly the committed turn on reset. Before this
fix, `reset_turn` erased `turn_audio_start` bytes, which is 0 on a normal turn,
so the audio of every turn accumulated. Multi-turn ASR then saw the whole
session and could return empty text.

### Echo-robust VAD and double-talk (bounded DSP slice)

With `server_vad` and `audio_output` enabled, every accepted `audio.delta` also
feeds that realtime session's far-end reference ring. This is PCM **accepted for
WebSocket emission**, not a measurement of the client's DAC, speaker, jitter
buffer, or acoustic path. No `playback_ack` or client change is required.
Reference-less sessions keep the previous energy-VAD behavior.

`src/serve/echo_robust_vad.hpp` processes each 10 ms near-end frame as follows:

1. Downmix far-end PCM16 to mono and linearly resample 8–96 kHz input to the
   16 kHz microphone contract. Invalid metadata or a mid-response format change
   clears the reference and falls back to energy VAD. Non-16 kHz resampling
   holds one input sample for interpolation.
2. Search integer delays from 0 through 500 ms. Acquisition screens every lag
   with 16 taps, then evaluates the best 16 candidates over all 160 samples. A
   sparse onset whose screen contains no energy uses the full frame for that
   same bounded lag scan. A trusted track (normalized correlation at least
   0.98) searches only its ±16-sample neighborhood; uncertain tracks reacquire.
3. At the best lag, require correlation at least 0.55 and reference energy above
   -60 dBFS, fit one signed gain (magnitude capped at 8), and subtract the
   aligned reference. A result that would amplify energy is discarded. Highly
   correlated echo with residual/input energy at most 0.20 is gated as echo,
   including moderate nonlinear clipping. Otherwise the residual goes to the
   existing adaptive `EnergyVad`. Independent near-end energy therefore remains
   available for double-talk interruption.
4. If there is no reference, too little history, inadequate correlation, or no
   in-window alignment, pass the original frame unchanged. There is no guessed
   alignment and no learned-model claim.

The bounds are explicit. Defaults retain 16,000 float reference samples (one
second, 64,000 bytes) plus two 160-sample work vectors and one PCM frame, about
64 KiB of sample/frame storage per session. The default worst case is 8,050
candidate evaluations per 10 ms frame: 33 local + 8,001 acquisition + 16
full-frame finalists. Steady trusted tracks use 33. There is no per-frame heap
allocation in the alignment/residual path and no history growth with response
length. The existing 64-session server cap bounds aggregate session state.

Reference state is per WebSocket session. It is cleared at a new response,
barge-in, timeout/shutdown, and configuration replacement, so purged or old
PCM cannot suppress speech against a later response. A normally completed
response keeps only its bounded tail until the next response boundary. The
original microphone PCM, not the VAD residual, is submitted to ASR.

`turn_detection.echo_suppression` is an optional boolean A/B control. It
**defaults to true**, and the default `transcription_session.updated` wire shape
is unchanged. Setting it to false stops feeding that session's emitted PCM to
the reference ring, so the existing `EnergyVad` sees the original microphone
signal; the effective reply explicitly reports `echo_suppression: false`.
While a turn or response is active, the only permitted configuration change is
this boolean with otherwise identical `server_vad` type and `silence_ms`. This
narrow exception lets the validation harness replay byte-identical microphone
PCM against the same active real-TTS response. It cannot change `playback_ack`,
response ownership, history, sampling, voice, or endpoint settings.

Deterministic CPU coverage in `tests/unified_contract_test.cpp` reports:

| Synthetic check | Result |
| --- | --- |
| Echo-only, 73 ms delay | 0 speech starts in 100 frames |
| WebSocket echo-only path | 0 false barge-ins in 50 frames; no response cancellation |
| Near-end-only and simultaneous double-talk | first 10 ms frame in both cases |
| Echo after near-end speech stops | configured 100 ms endpoint still fires |
| Delay jumps, 13–499 ms | 0 speech starts in 1,000 frames (10 s) |
| Attenuation 0.08/0.35, gain 1.4, moderate clipping | 4/4 echo-only cases produce no turn |
| 550 ms delay, outside the search bound | energy fallback detects in 10 ms; no false echo claim |
| Silence, repeated barge-in, response reset, two sessions | no silence event; 3/3 interruptions; no cross-response/session reference |
| 24 kHz far-end to 16 kHz near-end | resampled echo suppressed; final alignment valid |

These are deterministic digital fixtures, not acoustic quality measurements.
In particular, quiet double-talk below the residual gate, reverberation that a
single gain cannot represent, severe nonlinear distortion, client playout lead
or lag beyond the bounded history/search, and unrelated loud noise can still
cause a missed or false interruption.

**Required physical live gate — `NOT_RUN`.** Before
claiming physical AEC or deploying this as acoustic protection, run real TTS
through the actual speaker and microphone on each supported client topology.
Cover device-buffer delay and jitter, speaker/mic gain, clipping, silence,
echo-only, near-end-only, simultaneous speech, repeated barge-in, two sessions,
and response replacement. Record false barge-ins, missed double-talk, onset and
stop latency, endpoint latency, CPU, and resident memory against an A/B baseline
with the wrapper disabled. The gate requires zero echo-only cut-ins in the
agreed fixed corpus, no near-end-only regression, genuine double-talk stop
latency within the existing one-second L1 bound, and bounded resource use. A
human listening check is also required for playback artifacts and speech
intelligibility. The synthetic suite and timed virtual PCM harnesses do not
substitute for this gate.

### Playback acknowledgement and barge-in history (Stage B)

Server emission means **delivered to the client socket**, not played through
the client's speaker. A client that wants interrupted-turn continuity enables
the explicit opt-in during `session.update`:

```json
{
  "type": "session.update",
  "session": {
    "input_audio_format": "pcm16",
    "turn_detection": {"type": "server_vad", "silence_ms": 400},
    "audio_output": {"voice": "vivian", "playback_ack": true}
  }
}
```

The accepted `transcription_session.updated` echoes `playback_ack: true`.
Except for the isolated `echo_suppression` A/B toggle above, configuration
changes are rejected while a turn or response is active. That toggle never
changes acknowledgement state, so the mode cannot change between emission,
playback acknowledgement, and cancellation.
Leaving `playback_ack` absent or false preserves the old wire behavior: no playback ack is
required, ordinary completed realtime turns can enter history at completion,
and cancelled or failed turns never do. `realtime_playback_ack` is an internal
`UnifiedEngine` submission flag set only by this WebSocket path; it is not a new
public HTTP chat option and does not change HTTP cancellation/error exclusion.

In acknowledgement mode, response events carry a `response_id`. The added
cursor fields are UTF-8 **byte offsets** into that response's complete assistant
text, not Unicode codepoint counts or audio durations:

- `response.text.delta` has an informational `text_cursor`. Text emission alone
  never authorizes a playback acknowledgement.
- `audio.delta` has `text_cursor` and `text_end_cursor`. The first is the start
  of the current TTS sentence leg and therefore a conservative, already
  covered prefix; the second describes that leg's end. This deliberately does
  not claim character-level alignment inside a partially played sentence.
- After consuming the PCM through its actual playback path, the client sends
  `{"type":"output_audio_buffer.playback_ack","response_id":"...",
  "text_cursor":N}`. The server accepts only monotonically increasing cursors
  it issued on `audio.delta` or `audio.done` for that same response. Late,
  foreign, unknown, and repeated cursors are harmless and cannot extend the
  retained prefix.
- `response.playback_ack` echoes the accepted cursor and a `final` boolean.
  `audio.done` and `response.completed` mean that generation/emission finished,
  not that playback finished. Their final cursor becomes valid only after all
  PCM is consumed; that final ack records the complete turn and releases the
  retained terminal record.

A new utterance stops current playback and emits `response.cancelled` before
any new response events. The client must immediately stop playing that response
and ignore any already delivered late frames with its id. The event contains
`reason: "barge_in"` plus, in acknowledgement mode, `text_cursor`,
`acknowledged_text`, and `history_recorded`. The server removes queued old
`audio.delta`, text, completion, and ack frames. It retains the interrupted user
turn and exactly the acknowledged assistant prefix once: a pre-VLM cancellation
retains an empty assistant prefix, partial TTS never contributes its unplayed
suffix, and repeated cancellation cannot append or lengthen the committed turn.
History remains isolated by realtime session id.

The exception is intentionally narrow. Only an eligible realtime request with
playback acknowledgement and `session_history` can use it. Timeout, shutdown,
ordinary `/abort`, ordinary realtime cancellation, and every failed request keep
the default exclusion. A generation-complete response awaiting playback can be
interrupted without upgrading its emitted full answer into history.

The deterministic contract suite covers pre/post-VLM barge-in, partial TTS,
late audio/text, stale and repeated cancellation/acknowledgement, final
playback commit, exact next-prompt roles/content, default cancel/error
exclusion, and two-session isolation.

### CLI physical-device client (offline contract verified)

`umm-infer voice` is the opt-in native CLI counterpart to the web/macOS voice
controllers. It captures exact 16 kHz mono PCM16, negotiates
`audio_output.playback_ack=true`, streams `input_audio_buffer.append`, and plays
`audio.delta` through an isolated native output queue. The legacy
`umm-infer realtime --file ...` command and all chat/image/video commands keep
their existing behavior.

The CLI acknowledges a cursor only after the actual output-completion boundary,
not when a frame arrives or is queued. Its ACK worker preserves consumption
order, suppresses duplicate/decreasing cursors, and delays `audio.done`'s final
cursor until every queued PCM chunk drains. A speech-start edge immediately
invalidates local playback; `response.cancelled`, repeated cancellation and late
device callbacks are idempotent. Stop Voice, stdin EOF, signals and reconnect
stop both device paths and close the old socket. Reconnect creates a fresh
session and never replays buffered microphone input or old ACKs.

Linux uses runtime-loaded ALSA, with exact-format capture and hardware-delay
playback accounting. macOS uses AudioQueue input and output-completion
callbacks. No physical microphone, speaker, listening, macOS, or live-server
validation was run for this CLI implementation; the exact manual checklist and
platform limits are in [client.md](client.md#full-duplex-voice-voice). The
deterministic fake-WebSocket/fake-audio suite is:

```sh
umm-test-local voice-test
```

It covers cursor monotonicity, delayed/final ACKs, partial consumption,
cancellation races, repeated cancellation, stale callbacks, reconnect, cleanup,
and error paths. It is contract evidence, not a listening test or a real-model
claim.

### Deterministic CPU concurrency, cancellation and lifecycle stress gate

`tests/realtime_stress_test.cpp` drives the **production** `VoiceClient` and
`VoicePlayback` implementations. Only sockets, microphones and output devices
are scripted; the production connection thread, ACK worker, cancellation epoch,
queue accounting, reconnect and teardown code all run. A SplitMix64 permutation
selects the next session/action. Socket actions complete at explicit scheduled
boundaries, and asynchronous ACK assertions wait for state rather than assuming
a thread schedule. Failures print the exact seed, iteration, round, session,
stage and the last 32 actions. `--trace` prints the complete action schedule.

The default native test suite includes three iterations of six isolated voice
sessions (252 scheduled actions), with echo protection enabled and disabled on
alternate sessions and an active A/B toggle. Run the complete local suite plus
the named target with:

```sh
umm-test-local realtime-stress-test
```

Run just the dedicated target on this Linux dependency layout with:

```sh
zig build realtime-stress-test \
  -Ddeps-prefix="$HOME/.local/opt/umm-deps" \
  -Dcurl-prefix="$HOME/.local/opt/curl-ws"
```

Its expected summary is:

```text
REALTIME_STRESS PASS seed=6221112 iterations=3 sessions=6 scheduler_actions=252 echo_modes=on,off queue_bound=4096 trace=0xb2066d415e934bee
```

For the recorded repeated soak, the following exact command was run twice. Both
runs passed 3,360 scheduled actions and produced byte-identical logs with trace
`0x4ce990317a0823eb` (`cmp` exit 0):

```sh
zig build realtime-stress-test \
  -Ddeps-prefix="$HOME/.local/opt/umm-deps" \
  -Dcurl-prefix="$HOME/.local/opt/curl-ws" -- \
  --seed=0xdecafbad --iterations=20 --sessions=12
```

The per-session scenario covers tagged microphone chunks and ASR finals, VLM
text, far-end PCM, out-of-order and repeated device completions, monotonic and
duplicate/decreasing playback ACKs, duplicate terminal frames, local and server
barge-in cancellation, stale text/audio/ACK delivery, reconnect without mic/ACK
replay, and Stop Voice with a device callback still in flight. A separate case
fills the production 4,096-item playback queue, requires the next push to fail
closed, cancels it without an ACK, and blocks an ACK sink while `close()` runs.
Every session's text, PCM, response id, request cursor and completion callback is
checked against that session's tag.

`test_realtime_deterministic_coordinator_history_stress` in
`tests/unified_contract_test.cpp` independently exercises the real Coordinator,
ZMQ control messages and `UnifiedEngine`: six sessions times three rounds,
seed `0x38c001d`, 18 ASR -> VLM -> two-sentence streamed-TTS requests. Fixed
permutations interleave submissions, ASR finals, VLM/TTS deltas, final ACKs,
barge-in, duplicate backend terminals and late backend work after release. It
also saturates the test-only six-request admission limit and rejects one extra
submission without reaching ASR. The expected summary is:

```text
REALTIME_COORDINATOR_STRESS PASS seed=59506717 sessions=6 rounds=3 requests=18 history_exact=1 terminal_double=0 stale_after_release=0
```

Every next-turn prompt is compared with an independent per-session role/content
model. Cancelled turns contain exactly the acknowledged assistant prefix;
completed turns enter history only after the full cursor; repeated cancellation
and final ACKs do not duplicate or lengthen it. No late text/PCM, cross-session
event, reused request id or double terminal is accepted.

The stress exposed two real callback-lifecycle defects: a duplicate
`response.completed` invoked the user completion callback twice, and a repeated
`response.cancelled` repeated barge-in telemetry/callbacks.
`src/client/voice.cpp` now keeps bounded per-connection terminal-event
tombstones. This changes no public wire field, opt-in default, playback cursor or
server history rule. The new stress case is the red/green regression for both.

Recorded sanitizer commands on 2026-11-26:

```sh
# ThreadSanitizer: dedicated realtime client/playback target, PASS.
zig build realtime-stress-test -Dtsan=true \
  -Ddeps-prefix="$HOME/.local/opt/umm-deps" \
  -Dcurl-prefix="$HOME/.local/opt/curl-ws" -- \
  --seed=0x5eed38 --iterations=3 --sessions=6

# GCC AddressSanitizer + UndefinedBehaviorSanitizer + leak detection, PASS.
tests/realtime_stress_sanitize.sh voice -- \
  --seed=0x5eed38 --iterations=3 --sessions=6

# Optional full-source ASan/UBSan driver (not completed on this host).
tests/realtime_stress_sanitize.sh unified
```

Set `UMM_DEPS_PREFIX`, `UMM_CURL_PREFIX`, `UMM_SANITIZE_BUILD_DIR` or `CXX` when
the defaults do not match the host. The dedicated voice ASan/UBSan command
passed with leak detection and halt-on-error. The optional full GCC
ASan/UBSan compilation exceeded 600 seconds before producing a test binary and
was stopped, so Coordinator ASan/UBSan coverage is `NOT_RUN` here; the ordinary
full native suite passed separately.

The full `unified_contract_test` Zig TSan
configuration is **not claimed as covered on this host**: its instrumented
binary segfaulted before producing a test or TSan diagnostic, while the dedicated
realtime TSan binary above passed. This is recorded as an unsupported full-suite
TSan configuration, not as a passing or failing product race result.

These are deterministic digital contract/stress results. They do not establish
real-network timing, real model output, physical speaker/microphone echo
behaviour, listening quality, or live full-duplex audio quality. No GPU, model
worker, deployment or resident service is used. The physical-device/live gates
above remain separately required.

### Real public WebSocket/network and native-model gate (SM90, 2026-11-26)

This gate complements, and does not duplicate, the deterministic CPU stress above.
Implementation commit `8994399a7b8f125e0e6459cba0708276ebe3a325` integrated the
accepted deterministic stress non-destructively with newest main `7687efa`. The
real path was public loopback `WS /v1/realtime` -> native Qwen3-ASR-1.7B on
physical GPU 0 -> native Qwen3.8-27B and Qwen3-TTS-12Hz-1.7B-CustomVoice sharing
physical GPU 1. GPU 7 was reserved and unused; the resident service and GPUs 2-6
were not contacted. Model revisions stayed `UNKNOWN`.

```sh
python3 -I tests/realtime_network_test.py
python3 -I tests/realtime_network_live.py \
  --bin-dir "$UMM_BIN_DIR" --asr-model "$UMM_ASR_MODEL" \
  --vlm-model "$UMM_VLM_MODEL" --tts-model "$UMM_TTS_MODEL" \
  --audio "$UMM_ASR_AUDIO" --out "$UMM_OUT_DIR" --log-dir "$UMM_LOG_DIR" \
  --source-commit 8994399a7b8f125e0e6459cba0708276ebe3a325 \
  --seed 0x41c0de --rounds 3
python3 -I tests/realtime_network_verify.py "$UMM_OUT_DIR" --write
```

The gateway used pinned Zig 0.16.0; native stages used CUDA 12.4.131 for SM90.
Only new task-owned loopback processes were started. `run.json` records their
pids, wall/monotonic start times and Linux process-start ticks; teardown checks
those identities before signalling them. Both assigned GPUs were idle at initial
and immediate-launch preflight and returned to 3 MiB, 0%, no compute process.
All four owned processes exited, isolated IPC/SHM directories were removed, and
`/health` ended at `in_flight=0`.

| Independent raw-frame result | Value |
| --- | ---: |
| Concurrent isolated sessions / repeated rounds | 6 / 3 |
| Real WebSocket connections (including reconnects) | 15 |
| Connection-scoped response-ID entries (30 server responses + 11 ACK-only non-server entries) | 41 |
| Exact next-turn native VLM prompts checked / actual server responses | 30 / 30 |
| Completed/cancelled history turns checked | 21 |
| User terminal callbacks (exactly once per callback kind) | 27 |
| Stale events / cross-session leaks / duplicate callback kinds | 0 / 0 / 0 |
| Accepted decreasing cursors / final in-flight | 0 / 0 |
| Evidence files verified by `SHA256SUMS` | 44 |

The retained verifier's original `responses=41` is a connection-scoped count of
RX and TX response identities, not 41 real ASR -> VLM -> TTS responses. Its 11
non-server entries comprise eight foreign-ACK-only connection entries and three
stale old-response ACK entries after reconnect. There are 38 globally unique
response IDs and 30 actual server responses, matching all 30 native VLM prompts.
The independent CPU timing audit re-decodes and reconciles this distinction,
all exact timings, ACK/cancel schedules, PCM hashes, prompts, history, resource
samples, and outliers without changing the runtime; see
[realtime-latency-audit.md](realtime-latency-audit.md).

Each scenario occurred three times: full final ACK, immediate barge-in, delayed
barge-in after an acknowledged sentence prefix, reconnect during playback,
reconnect during in-flight generation, and teardown with an active timer callback.
Seeded round permutations, per-session jitter and ACK delays are retained in
`plan.json`. The driver also sent duplicate, decreasing, foreign and unissued ACKs,
performed late old-response ACKs after reconnect, and checked late text/audio and
terminal frames after cancellation. A generation-complete response awaiting
playback can legally emit one `response.completed` and later one
`response.cancelled`; the verifier rejects a duplicate of either callback kind
rather than weakening the protocol by accepting duplicates or forbidding this
legal sequence.

The echo A/B used the **same active real TTS response** and byte-identical near
and far-end PCM. With 73 ms echo, the protected arm had zero cut-ins and the
disabled arm had one cancellation. Microphone SHA-256 was
`0ce82424419fc5aa5b8cf5294890772e45c5bdc3b49c56150b9799550cca698f`;
far-end SHA-256 was
`85be5639dab889cea9da1740f2ab392567b78abedcfc969d5c3bb230eef0750e`.
Simultaneous double-talk cancelled in both echo modes and retained the exact
acknowledged prefix. Double-talk necessarily moves to subsequent real responses
after the first cancellation; no false claim is made that two different TTS
responses have identical far-end bytes. Physical playback/capture is not involved.

The raw evidence contains exact masked client and unmasked server wire frames,
strictly increasing per-connection monotonic arrival/transmit times, response and
request identities, every PCM chunk hash, captured PCM, ACK/cancel schedules, all
next-turn prompt observations, process logs and resource samples. A separate
fresh process decodes those wire bytes and the byte-preserving ZMQ forwarding
observer's raw submissions; it derives history independently rather than trusting
the driver. The production queue caps are pinned at 1,024 outbound frames, 4,096
partial events, 4,096 unified events and 64 MiB audio per session. Server RSS grew
21,348,352 bytes, below the 268,435,456-byte gate; deterministic stress separately
covers queue saturation. No real-server queue-saturation claim is made.

Whole-model manifest SHA-256 values (including every model file) were ASR
`a16807fc65e05f36687484bdfb8cb9f103c3fb59f6787d1f904f3ffebf8a3005`, VLM
`fb2a8751df1dfbcdf556988acd8e2ac7c6f7180c609797d5e9b3a8e3e40a1c7e`, and TTS
`92db72616c9578462add19ed81720b7abf1c1e9c3f4d225877a1cd6104d2fb27`.
Binary SHA-256 values were gateway
`e7f569bcd848ae42dec221d2b9c0b9c950ca98d8100aa366a245a243fa351e83`, ASR
`3c501baabf3302859a1333c9cfc38a357e4072b8d5bcffe85fcf6e8f8ffa370e`, VLM
`caf01c8dca9e33d9fcb30d23fc9bcd9e8e51abaf15f673b578e0133a4f6c3159`, and TTS
`8906fb7c852b778c336ad03ae402ded84a62fac7318c528c27dc5223a4baba6e`.
Harness and scheduler/verifier SHA-256 values were respectively
`8b0be64d03db09ac8723e7676a30fb83c769726492259441573ddc1829eb589a` and
`90424bf598d2c8c3fbc1eae666fff90af67bc034298d0c5da877f045bd9bef0a`.
`run.json`, `independent-verification.json`, and `SHA256SUMS` SHA-256 values were
`5d01adb537008856302f455a6b3667fad723b1d30623de7827d3688a9c8ab427`,
`30cf8077ab7fa7bafcdc53d030213dc8abca603adc6b14c706968957465a9029`, and
`a06aaef64e7726d67a8652fa6522b013872d39e0c4bf97d72fcf8f22a2d9b355`.
The 44-file manifest passed again after retrieval. Raw evidence is an uncommitted
run artifact under `$UMM_OUT_DIR`; two earlier driver-only failed attempts are not
counted as passes, and their owned-process/GPU cleanup also succeeded.

This is **REAL_DIGITAL_LOOPBACK_NETWORK_AND_NATIVE_MODELS** evidence only. It is
not WAN, physical microphone/speaker, acoustic echo-cancellation, perceptual
listening, or generalized performance evidence. Those physical and perceptual
checks remain `NOT_COVERED`.

### Incremental prefill

The omni stage keeps one prefix cache per process: the session id, the prompt
ids of the last request, and how many of its KV slots came from prefill. A
request in the same session reuses `min(longest common prefix, cached prefill
length)` tokens, embeds only the suffix, and prefills it with `history=prefix`
and the position ids sliced from the full prompt. Only prefill-written slots
are reusable; slots written by decode are not, because decode uses a different
reduction path.

The cache is on by default; `UMM_OMNI_PREFIX_CACHE=0` disables it. The stage
renders every message of the request, so `session_history` is part of the
prompt instead of being dropped. The stage reports `prefix_reused`,
`prefill_tokens` and `prefill_ms` per request.

## Measured on SM90, 2026-11-17 (L2)

Environment: SM90 GPU container, native ASR on GPU 2, native Omni thinker on
GPU 7, native Qwen3-TTS on GPU 5, local model directories (`$UMM_MODELS_DIR/{asr,omni,qwen3-tts}`), real speech
`asr_en_pcm16.wav`, server and stages built from this commit. Ports were
ephemeral; the resident services and GPU 0/1 were not touched.

Command, inside the container:

```sh
python3 -I tests/full_duplex_l2_live.py --mode all \
  --bin-dir bin --asr-model .../models/asr --omni-model .../models/omni \
  --tts-model .../qwen3-tts --audio .../asr_en_pcm16.wav \
  --out evidence/full --log-dir logs/full \
  --gpu-asr 2 --gpu-vlm 7 --gpu-tts 5 --turns 20
```

Raw evidence: `evidence/rt-fix/l2-evidence.json` for the realtime mode,
`evidence/fix2/l2-evidence.json` for the pipeline mode,
`evidence/prefix/l2-evidence.json` for the prefill measurement (the first-turn
prompt is longer there, so the shared prefix is measurable), and the stage logs
next to them. They are not committed artifacts.

The realtime and pipeline modes were rerun after the streaming waveform change
in commit `291a540`, with the same command and `--mode realtime` or
`--mode pipeline`. The prefix evidence predates that change: the prefill code
was not touched, so `evidence/prefix` is reused and was not rerun.

### First audio, 20 spoken turns over one WebSocket

Each turn streams 2 s of real speech in 100 ms chunks, then 600 ms of silence;
`silence_ms` is 400. The measurement starts at the last speech chunk and ends
at the first `audio.delta`.

| Metric | Value |
| --- | --- |
| samples | 20 |
| p50 | 771.9 ms |
| p95 | 954.3 ms |
| min / max | 716.8 / 1033.6 ms |

The same run produced 157 streamed TTS legs. Every one logged
`tts stream_check identical=1`: the chunks concatenated in order equaled the
terminal WAV bit for bit, with no tolerance.

### Sentence pipeline advance

Same prompt, two HTTP requests with `stream=true` and audio output. In
`sentence_stream` the request sets `audio.stream=true`; `single_leg` omits it,
so one TTS leg starts after the VLM answer ends.

| Metric | sentence_stream | single_leg |
| --- | --- | --- |
| submit to first audio | 302.1 ms | 23091.0 ms |
| last VLM delta to first audio | -317.8 ms | 22507.9 ms |
| first audio before last VLM delta | true | false |
| audio chunks | 45 | 1 |

End-to-end first-audio advance: **22825.7 ms**.

The single-leg baseline synthesizes the whole answer before its first sample,
by construction. In this run the one-shot leg produced 512 codec frames (the
stage cap) for the same text, so its 22.5 s is what that path does for this
input, not a typical answer length. The comparison is end-to-end and includes
that cost. The streaming run's first audio arrived 317.8 ms before the last VLM
delta.

### Incremental prefill, two-turn session with `session_history`

The same two-turn conversation runs once with `UMM_OMNI_PREFIX_CACHE=1` and
once with `=0`. Stage-reported numbers for the second turn:

| Metric | cache off | cache on |
| --- | --- | --- |
| second-turn prefill | 127.9 ms | 88.4 ms |
| reused prompt tokens | 0 | 71 |
| answers identical | - | true |

Prefill time reduction: **30.9%**. The cached and
uncached answers are compared byte for byte and are equal.

## Measured on SM90, 2026-11-25 (Stage B barge-in history)

Environment: implementation commit `b392e34`, native Qwen3-ASR-1.7B on GPU 0,
native Qwen3.8-27B (`qwen38`) on GPU 1, and native
Qwen3-TTS-12Hz-1.7B-CustomVoice on GPU 7. Model revisions stayed `UNKNOWN`.
Each selected GPU was checked immediately before the run: 3 MiB used, 0%
utilization, and no compute process. HTTP/WebSocket ports were ephemeral and
the resident service on 43301 and GPUs 2-6 were not touched. The gateway was
built in an isolated directory; its SHA-256 was
`5f494e559652fd352fc209645e6df5b8768d261954af5aef086c4fd5b71d82dc`.

Command, with real paths supplied through the named environment variables:

```sh
python3 -I tests/full_duplex_barge_in_live.py \
  --bin-dir "$UMM_BIN_DIR" --asr-model "$UMM_ASR_MODEL" \
  --vlm-model "$UMM_VLM_MODEL" --tts-model "$UMM_TTS_MODEL" \
  --audio "$UMM_ASR_AUDIO" --out "$UMM_OUT_DIR" --log-dir "$UMM_LOG_DIR" \
  --gpu-asr 0 --gpu-vlm 1 --gpu-tts 7
```

The committed harness uses timed virtual PCM consumption, not a physical audio
device: it sleeps for each chunk's actual sample duration and only then sends
its cursor. Raw evidence is
`$UMM_OUT_DIR/full-duplex-barge-in-evidence.json`; stage logs are under
`$UMM_LOG_DIR`. These are uncommitted run artifacts.

| Check | Result |
| --- | --- |
| GPU preflight and cleanup | PASS; GPUs 0/1/7 started and ended at 3 MiB, 0%; no task process remained |
| Real ASR turns | `Uh huh. Oh yeah, yeah.` before and after barge-in |
| Acknowledged assistant prefix | 12-byte UTF-8 prefix `Sounds like `, exactly matching the consumed cursor; the remaining generated answer was excluded |
| First barge-in stop latency | 400.8 ms (the fixture begins with 400 ms of digital silence) |
| Following VLM response | PASS; new response id `req-2` started after the post-barge-in transcript |
| Repeated barge-in stop latency | 400.8 ms |
| Late/stale old-response output | 0 events after either `response.cancelled` |
| Final queue state | `/health` returned `{"status":"ok","in_flight":0}` |
| Overall harness verdict | `PASS` |

This validates the protocol and real model cancellation path, including a
partially consumed multi-sentence answer. It is not a listening test and does
not establish acoustic text alignment inside a TTS sentence.

## Measured on SM90, 2026-11-26 (real echo/double-talk PCM gate)

This is **real-model server/PCM validation**, not physical acoustic echo
cancellation proof. Implementation commit `97d77b4bb5bade11e7ed38d2711eafd6e450505f`
used native Qwen3-ASR-1.7B on physical GPU 0, Qwen3.8-27B on GPU 1, and
Qwen3-TTS-12Hz-1.7B-CustomVoice on GPU 7. Model revisions stayed `UNKNOWN`;
config SHA-256 values were respectively
`2e74a751548b8ad7d7526d29365ad8144c345d8b412b1152d25dc6698452712f`,
`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`, and
`17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9`.
The ASR and VLM indexes identified 2 and 18 distinct shards. The resident
service on 43301 and GPUs 2-6 were not used.

```sh
python3 -I tests/full_duplex_echo_live.py --self-check
python3 -I tests/full_duplex_echo_live.py \
  --bin-dir "$UMM_BIN_DIR" --asr-model "$UMM_ASR_MODEL" \
  --vlm-model "$UMM_VLM_MODEL" --tts-model "$UMM_TTS_MODEL" \
  --audio "$UMM_ASR_AUDIO" --out "$UMM_OUT_DIR" --log-dir "$UMM_LOG_DIR" \
  --gpu-asr 0 --gpu-vlm 1 --gpu-tts 7 --source-commit "$UMM_SOURCE_COMMIT"
```

The fixed corpus contains six echo-only cases. Each protected arm captures real
`audio.delta` PCM and constructs a deterministic 16 kHz microphone stream with
500 ms of active echo followed by 600 ms of digital silence. Its oldest
referenced sample is exactly 500 ms old (490 ms including the current 10 ms
frame), leaving delay headroom inside the production search window. The disabled
arm then replays the **same bytes against the same still-active real response**
with `echo_suppression=false`. Thus the A/B comparison does not depend on TTS
cross-request bit reproducibility. The recorded microphone hashes are distinct
across cases and bind both arms.

| Echo-only case | Protected cut-ins / ASR turns | Disabled cut-ins | Disabled cancel from first append | Endpoint from last above-floor frame |
| --- | ---: | ---: | ---: | ---: |
| 0 ms, gain 0.35 | 0 / 0 | 1 | 0.478 ms | 341.339 ms |
| 73 ms, gain 0.55 | 0 / 0 | 1 | 0.497 ms | 341.188 ms |
| 249 ms, gain 1.40 | 0 / 0 | 1 | 0.509 ms | 331.368 ms |
| 499 ms, gain 0.08 | 0 / 0 | 1 | 0.470 ms | 381.281 ms |
| 117 ms, 491 clipped samples | 0 / 0 | 1 | 0.445 ms | 331.368 ms |
| delay jumps 13/117/249/499/73 ms | 0 / 0 | 1 | 200.765 ms | 301.018 ms |

Every disabled arm emitted `speech_started`, cancelled the old response,
endpointed, and returned a real ASR final for the echo turn. Those transcripts
are evidence that the fixture actually crosses the old energy-VAD boundary, not
an ASR-quality result. Every protected arm had zero cut-ins, endpoints, and
transcription events. Each cancellation retained the exact acknowledged UTF-8
prefix; echo cases used a 10-byte proper prefix `It sounds ` and no stale output
followed cancellation.

| Additional real scenario | Result |
| --- | --- |
| Near-end-only A/B | Same microphone SHA-256 `2d8879b7cb71808f249434643225d6873496e2b7d3e03c7b6d546338d13e819f`; protected onset/cancel 3.355/3.347 ms, disabled 0.498/0.490 ms; identical final transcript and no onset regression |
| Simultaneous double-talk, two responses | 3.375 ms and 3.256 ms from input to `response.cancelled`, both within the one-second L1 bound; each ended with a real ASR final |
| Stale reference across responses | Response `req-17`'s echo cancelled `req-18` in 0.323 ms before `req-18` emitted any audio; empty acknowledged prefix stayed empty, proving the old ring was not reused |
| Two sessions | Session A's own echo caused 0 cut-ins; the same PCM on reference-less session B started once in 0.313 ms and endpointed without cancelling A |
| Server resource bound | p95 CPU 19.923%, maximum CPU 59.754%, maximum RSS 44,466,176 B, observed RSS growth 17,948,672 B; all inside the declared 200% / 512 MiB / 128 MiB bounds |
| Terminal state and cleanup | `/health` `in_flight=0`; all task processes exited; isolated IPC/SHM directories removed; GPUs 0/1/7 each ended at 3 MiB, 0%, no compute process |

Binary SHA-256 values were `umm-infer`
`6119b0ca5746576d644f7c594136820ad5c734a060c17adff87a3e15f4dfe348`, ASR
`4f56591272c4d76b92c32ccca4ab3191b46c40bce674652261eb8f4c34f5e22b`,
Qwen3.8 `eb6fd4dc4d47c8f2ecb64702e44dd38be23cb82987ff59da8ae84965f6fb2f38`,
and TTS `c4f9321cff7e50d8acacdb6dccc633a61d230da05d05015340186684a7e28782`.
Harness SHA-256 was
`e8171f4787568d9d85156ef653ac16b148569dc4b24d5e18a6b719a80ce0ab1b`.
Raw far-end PCM, every microphone stream, raw WebSocket payloads, client transmit
timing, 50 ms CPU/RSS samples, and all four server/stage logs are uncommitted
artifacts under `$UMM_OUT_DIR` / `$UMM_LOG_DIR`. `sha256sum -c SHA256SUMS`
verified all 63 files. The summary SHA-256 was
`030abef978faca8ed362f9d1e0dede8c1b302bdce570d2851e3e5ab73d93d78c`;
`SHA256SUMS` SHA-256 was
`39a770f36dd92a4ceecdf0be3141430fc82771bf2ff14543fdf6a71840e21154`.

Physical speaker/microphone topology, device delay/jitter, acoustic path,
playback artifacts, and listening quality remain **NOT_RUN**. This gate
establishes the real server and deterministic PCM contract only; it does not
satisfy the physical gate below or justify an acoustic-AEC claim.

## What L2 still needs

**Incremental ASR encoding.** Preview transcription is bounded to a 15-second
sliding window (`kMaxPreviewSpanBytes` in `src/serve/realtime_api.cpp`) so
previews stay O(1) on long monologues, while final commit still transcribes the
full turn. A streaming encoder with cached state would avoid re-encoding the
preview window altogether.

**Physical client playback validation.** The server has the explicit
cancellation and playback-acknowledgement cursor contract, and the live harness
exercises timed virtual PCM playback. The Web UI and the opt-in `umm-infer voice`
CLI now implement microphone capture, consumption-driven playback ACKs and
immediate `response.cancelled` stop. Their offline suites mock the audio
boundary; neither suite validates a physical microphone, speaker, or live
server. Physical-device execution and listening remain `NOT_RUN` for the CLI in
this environment, so audible quality, device latency, echo behavior and
ALSA/AudioQueue hardware compatibility are not established.
Character/acoustic alignment inside one TTS sentence also remains unsolved:
retention is conservative at sentence boundaries rather than estimating an
unverified substring from PCM duration.

**More than one cached session.** The omni stage caches one session's prefix.
Concurrent sessions each need their own KV snapshot.

## What L3 still needs

L3 is turn semantics learned from data. L1 endpoints on silence, so a thinking
pause ends a turn. L3 needs:

- a semantic end-of-turn model over partial transcripts and audio, not only
  energy;
- overlap handling: both sides speak, and the system decides whether to yield;
- a training corpus of real multi-turn conversations with barge-in labels;
- evaluation of turn-taking latency and false cut-ins, not only WER.

Nothing in this commit trains or evaluates such a model.
