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
  energy detector with an adaptive noise floor; it has no new dependency.
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
- No echo cancellation. A speaker playing through the client's own output can
  start a turn.
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
Configuration changes are rejected while a turn or response is active, so the
mode cannot change between emission, playback acknowledgement, and cancellation.
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

## What L2 still needs

**Incremental ASR encoding.** Preview transcription is bounded to a 15-second
sliding window (`kMaxPreviewSpanBytes` in `src/serve/realtime_api.cpp`) so
previews stay O(1) on long monologues, while final commit still transcribes the
full turn. A streaming encoder with cached state would avoid re-encoding the
preview window altogether.

**Physical client playback.** The server now has the explicit cancellation and
playback-acknowledgement cursor contract, and the live harness exercises timed
virtual PCM playback. The Web UI has an opt-in realtime mode that captures
16 kHz PCM16 microphone frames, plays `audio.delta` through Web Audio, sends
playback ACKs from `onended`/final queue drain, and stops scheduled sources on
`response.cancelled`; its offline contract suite mocks the audio nodes and does
not validate a physical microphone, speaker, or live server. The bundled CLI
client still does not play PCM16 through an audio device or stop a physical
output stream on `response.cancelled`. Character/acoustic alignment inside one
TTS sentence also remains unsolved: retention is conservative at sentence
boundaries rather than estimating an unverified substring from PCM duration.

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
